# %% [markdown]
# # MultiREx Autoencoder for Exoplanet Spectra Denoising (H2O)
# This script trains a deep learning model to remove noise and stellar contamination
# from exoplanet transit spectra. This version is fully optimized for memory efficiency
# on both CPU and GPU.

# %% [markdown]
# ## 1. Initial Setup and Imports

# %%
import multirex as mrex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import gc
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% [markdown]
# ## 2. Configuration and Helper Functions

# %%
def remove_warnings():
    """Suppresses specified warnings for cleaner output."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Pandas doesn't allow columns to be created via a new attribute name*",
    )
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def configure_gpu_memory_growth():
    """Prevents TensorFlow from allocating all GPU memory at once."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow memory growth set for {len(gpus)} GPU(s).")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"GPU memory growth could not be set: {e}")

# Initial setup
remove_warnings()
configure_gpu_memory_growth()

# Load and prepare wavelength data
waves = np.loadtxt("waves.txt")
n_points = len(waves)
wn_grid = np.sort((10000 / waves))


def apply_contaminations_from_files(contamination_files, df, n_points):
    """
    Applies multiple stellar contaminations to the spectral data.
    This version is optimized to prevent DataFrame fragmentation.
    """
    df_list = []
    # Add non-contaminated case
    df_no_contam = df.copy().assign(f_spot=0.0, f_fac=0.0)
    cols = ["f_spot", "f_fac"] + [col for col in df.columns if col not in ["f_spot", "f_fac"]]
    df_list.append(df_no_contam[cols])
    
    pattern = r"fspot(?P<f_spot>[0-9.]+)_ffac(?P<f_fac>[0-9.]+)\.txt$"

    for file_path in contamination_files:
        if not os.path.isfile(file_path): raise FileNotFoundError(f"File not found: {file_path}")
        
        filename = os.path.basename(file_path)
        match = re.search(pattern, filename)
        if not match: raise ValueError(f"Filename '{filename}' does not match pattern.")

        f_spot = float(match.group("f_spot"))
        f_fac = float(match.group("f_fac"))

        try:
            contam_data = np.loadtxt(file_path, ndmin=2)
            contam_values = contam_data[:, 1] if contam_data.shape[1] >= 2 else contam_data.flatten()
            if len(contam_values) != n_points: raise ValueError(f"Contamination values in '{filename}' != n_points.")
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")

        df_contam = df.copy()
        data_columns = df_contam.columns[-n_points:]
        df_contam[data_columns] *= contam_values[::-1]
        df_contam = df_contam.assign(f_spot=f_spot, f_fac=f_fac)
        df_list.append(df_contam[cols])

    df_final = pd.concat(df_list, ignore_index=True).copy()
    df_final.data = df_final.iloc[:, -n_points:]
    df_final.params = df_final.iloc[:, :-n_points]
    return df_final

def filter_rows(df):
    """Filters rows based on atmospheric composition values."""
    filter_cols = ["atm CH4", "atm O3", "atm H2O"]
    for chem in [col for col in filter_cols if col in df.columns]:
        df = df[df[chem] >= -8].copy()
    df.data = df.iloc[:, -n_points:]
    df.params = df.iloc[:, :-n_points]
    return df

def load_and_prep_data(filepath, n_points):
    """Reads a CSV and sets appropriate data types for memory efficiency."""
    df = pd.read_csv(filepath)
    df[df.columns[-n_points:]] = df[df.columns[-n_points:]].astype('float32')
    return df

def normalize_min_max_by_row(df):
    """Normalizes each row of a DataFrame to a [0, 1] range."""
    min_vals = df.min(axis=1)
    max_vals = df.max(axis=1)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    return (df.sub(min_vals, axis=0)).div(range_vals, axis=0)

def generate_df_with_noise_std(df, n_repeat, noise_std, seed=None):
    """Generates a new DataFrame by applying Gaussian noise to the spectra."""
    # This function now correctly handles dataframes without .data/.params attributes
    if hasattr(df, 'data') and hasattr(df, 'params'):
        df_params = df.params
        df_spectra = df.data.astype('float32')
    else:
        # Assumes the last n_points columns are spectral data
        df_params = df.iloc[:, :-n_points]
        df_spectra = df.iloc[:, -n_points:].astype('float32')
    
    if seed is not None: np.random.seed(seed)
    
    replicated_spectra_vals = np.repeat(df_spectra.values, n_repeat, axis=0)
    
    if isinstance(noise_std, (int, float)):
        noise = np.random.normal(0, noise_std, replicated_spectra_vals.shape).astype('float32')
    else:
        noise_array = np.array(noise_std, dtype='float32')
        noise_replicated = np.tile(np.repeat(noise_array[:, np.newaxis], n_repeat, axis=0), (1, df_spectra.shape[1]))
        noise = np.random.normal(0, noise_replicated, replicated_spectra_vals.shape).astype('float32')
        
    noisy_spectra = pd.DataFrame(replicated_spectra_vals + noise, columns=df_spectra.columns)

    replicated_params = pd.DataFrame(np.repeat(df_params.values, n_repeat, axis=0), columns=df_params.columns)
    
    target_len = len(replicated_params)
    new_cols_data = {
        "noise_std": np.full(target_len, noise_std) if isinstance(noise_std, (int, float)) else np.repeat(noise_std, n_repeat),
        "n_repeat": np.full(target_len, n_repeat)
    }
    new_cols_df = pd.DataFrame(new_cols_data)

    df_final = pd.concat([new_cols_df, replicated_params.reset_index(drop=True), noisy_spectra.reset_index(drop=True)], axis=1)
    df_final.data = df_final.iloc[:, -n_points:]
    df_final.params = df_final.iloc[:, :-n_points]
    return df_final

# %% [markdown]
# ## 3. Load and Process Source Data

# %%
contamination_files = [
    "stellar_contamination/TRAPPIST-1_contam_fspot0.01_ffac0.08.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.01_ffac0.54.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.01_ffac0.70.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.08_ffac0.08.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.08_ffac0.54.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.08_ffac0.70.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.26_ffac0.08.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.26_ffac0.54.txt",
    "stellar_contamination/TRAPPIST-1_contam_fspot0.26_ffac0.70.txt",
]

# Initialize variables to prevent NameError if try block fails
airless_data, CO2_data, H2O_data = None, None, None

try:
    # Load and apply contaminations
    temp_airless = load_and_prep_data("spec_data/airless_data.csv", n_points)
    airless_data = apply_contaminations_from_files(contamination_files, temp_airless, n_points)

    temp_CO2 = load_and_prep_data("spec_data/CO2_data.csv", n_points)
    CO2_data = apply_contaminations_from_files(contamination_files, temp_CO2, n_points)
    
    # Load H2O data instead of CH4
    temp_H2O = filter_rows(load_and_prep_data("spec_data/H2O_data.csv", n_points))
    H2O_data = apply_contaminations_from_files(contamination_files, temp_H2O, n_points)
    
    # Clean up temporary dataframes
    del temp_airless, temp_CO2, temp_H2O
    gc.collect()

except Exception as e:
    print(f"Error during data loading: {e}")

# %% [markdown]
# ## 4. Generate Full Dataset with Caching

# %%
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)
# Use H2O-specific filenames to avoid overwriting CH4 data
noisy_path = os.path.join(output_dir, "H2O_X_noisy_full_dataset.npy")
clean_path = os.path.join(output_dir, "H2O_X_clean_full_dataset.npy")

if os.path.exists(noisy_path) and os.path.exists(clean_path):
    print("Found cached data. Loading from disk...")
    X_noisy = np.load(noisy_path)
    X_no_noisy = np.load(clean_path)
    print("...Loading complete.")
else:
    print("Cached data not found. Generating full dataset...")
    
    list_of_noisy_arrays, list_of_clean_arrays = [], []
    snr_values = [1, 3, 6, 10, None] 

    for snr in snr_values:
        print(f"--- Processing SNR = {snr if snr is not None else 'inf'} ---")
        noise = mrex.generate_df_SNR_noise(df=CO2_data, n_repeat=1, SNR=snr)["noise"][0] if snr is not None else 0.0

        # Generate noisy data
        temp_CO2 = generate_df_with_noise_std(df=CO2_data, n_repeat=5000, noise_std=noise)
        temp_H2O = generate_df_with_noise_std(df=H2O_data, n_repeat=500, noise_std=noise)
        temp_airless = generate_df_with_noise_std(df=airless_data, n_repeat=5000, noise_std=noise)
        noisy_df = pd.concat([temp_CO2, temp_H2O, temp_airless], ignore_index=True)
        list_of_noisy_arrays.append(normalize_min_max_by_row(noisy_df.iloc[:, -n_points:]).values.astype("float32"))
        del temp_CO2, temp_H2O, temp_airless
        gc.collect()

        # Generate corresponding clean data (from the same source dataframes but with noise_std=0)
        clean_CO2 = generate_df_with_noise_std(df=CO2_data, n_repeat=5000, noise_std=0)
        clean_H2O = generate_df_with_noise_std(df=H2O_data, n_repeat=500, noise_std=0)
        clean_airless = generate_df_with_noise_std(df=airless_data, n_repeat=5000, noise_std=0)
        clean_df = pd.concat([clean_CO2, clean_H2O, clean_airless], ignore_index=True)
        list_of_clean_arrays.append(normalize_min_max_by_row(clean_df.iloc[:, -n_points:]).values.astype("float32"))
        
        # Clean up all temporary dataframes for this loop
        del noisy_df, clean_CO2, clean_H2O, clean_airless, clean_df
        gc.collect()

    X_noisy = np.concatenate(list_of_noisy_arrays, axis=0)
    X_no_noisy = np.concatenate(list_of_clean_arrays, axis=0)
    del list_of_noisy_arrays, list_of_clean_arrays
    gc.collect()
    
    print("\n--- Saving data to disk for future runs ---")
    np.save(noisy_path, X_noisy)
    np.save(clean_path, X_no_noisy)

print(f"\nFinal noisy data shape: {X_noisy.shape}")
print(f"Final clean data shape: {X_no_noisy.shape}")
assert X_noisy.shape[0] == X_no_noisy.shape[0], "Sample count mismatch."

# %% [markdown]
# ## 5. Build and Train Autoencoder Model

# %% [markdown]
# ### Create Optimized TensorFlow Datasets

# %%
BATCH_SIZE = 64
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Create datasets on CPU to prevent VRAM overflow on initialization
with tf.device('/CPU:0'):
    X_train_noisy, X_test_noisy, X_train_clean, X_test_clean = train_test_split(
        X_noisy, X_no_noisy, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Create efficient TensorFlow dataset pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_noisy, X_train_clean))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test_noisy, X_test_clean))
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Clean up large NumPy arrays from RAM immediately after creating TF datasets
del X_noisy, X_no_noisy, X_train_noisy, X_train_clean
gc.collect()

print("TensorFlow datasets created and source NumPy arrays cleared from RAM.")

# %% [markdown]
# ### Define and Compile the Autoencoder

# %%
# Infer input dimension directly from the dataset specification
input_dim = validation_dataset.element_spec[0].shape[1]

input_spectrum = keras.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(512, activation="swish")(input_spectrum)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(512, activation="swish")(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(512, activation="swish")(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(300, activation="swish")(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(300, activation="swish")(encoded)

# Decoder
decoded = layers.Dense(300, activation="swish")(encoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(512, activation="swish")(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(512, activation="swish")(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(512, activation="swish")(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = keras.Model(inputs=input_spectrum, outputs=decoded)
optimizer = Adam(learning_rate=0.00001)
autoencoder.compile(optimizer=optimizer, loss="mae")

autoencoder.summary()

# %% [markdown]
# ### Train the Model

# %%
history = autoencoder.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
)

# Save the model with an H2O-specific name
autoencoder.save("AE_H2O.keras")
print("Model training complete and saved to AE_H2O.keras")

# %% [markdown]
# ## 6. Evaluate and Visualize Results

# %%
print("Plotting training history...")
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training MAE")
plt.plot(history.history["val_loss"], label="Validation MAE")
plt.title("Model MAE Progress During Training")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.legend()
plt.grid(True)
plt.show()

print("\nEvaluating model on test data...")
# Reload test data from cache for evaluation, as it was deleted to save RAM.
print("Loading test data for evaluation...")
# Use the H2O-specific cached filenames
noisy_path = os.path.join(output_dir, "H2O_X_noisy_full_dataset.npy")
clean_path = os.path.join(output_dir, "H2O_X_clean_full_dataset.npy")

X_noisy_full = np.load(noisy_path)
X_clean_full = np.load(clean_path)
_, X_test_noisy_eval, _, X_test_clean_eval = train_test_split(
    X_noisy_full, X_clean_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
del X_noisy_full, X_clean_full
gc.collect()


print("Predicting on test data...")
decoded_spectra = autoencoder.predict(X_test_noisy_eval, batch_size=BATCH_SIZE)

# Visualize a few reconstructions
print("Visualizing sample reconstructions...")
num_samples = 5
indices = np.random.choice(len(X_test_noisy_eval), num_samples, replace=False)

for idx in indices:
    plt.figure(figsize=(12, 5))
    plt.plot(waves, X_test_clean_eval[idx].flatten(), label="Original Clean Spectrum", color='blue')
    plt.plot(waves, X_test_noisy_eval[idx].flatten(), label="Noisy Input Spectrum", color='gray', alpha=0.6)
    plt.plot(waves, decoded_spectra[idx].flatten(), label="Denoised Spectrum", color='red', linestyle='--')
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Intensity")
    plt.title(f"Spectrum Reconstruction - Sample {idx}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# %% [markdown]
# ### Final Performance Metrics

# %%
print("\nCalculating final performance metrics...")
mae = mean_absolute_error(X_test_clean_eval, decoded_spectra)
print(f"Final Mean Absolute Error (MAE): {mae:.6f}")

mse = mean_squared_error(X_test_clean_eval, decoded_spectra)
print(f"Final Mean Squared Error (MSE): {mse:.6f}")

# Flatten for R² score
r2 = r2_score(X_test_clean_eval.flatten(), decoded_spectra.flatten())
print(f"Final Coefficient of Determination (R²): {r2:.6f}")

# %% [markdown]
# ## 7. Evaluate and Visualize Results (from Saved Model)
# This cell demonstrates the best practice for evaluation. It ensures that the model being tested
# is the exact one that was saved to disk, making the results reproducible.

# %%
from tensorflow import keras

# --- Step 1: Load the Saved Model ---
print("Loading the saved autoencoder model from AE_H2O.keras...")
saved_autoencoder = keras.models.load_model("AE_H2O.keras")
print("Model loaded successfully.")


# --- Step 2: Re-load Test Data for Evaluation ---
# (This part is the same as the previous cell, ensuring we have the correct test data)
print("\nLoading test data for final evaluation...")
# Define constants again to make this cell self-contained
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64

# Ensure 'noisy_path' and 'clean_path' are defined from the caching cell
noisy_path = os.path.join(output_dir, "H2O_X_noisy_full_dataset.npy")
clean_path = os.path.join(output_dir, "H2O_X_clean_full_dataset.npy")

X_noisy_full = np.load(noisy_path)
X_clean_full = np.load(clean_path)

# Perform the exact same train-test split to get the identical test set
_, X_test_noisy, _, X_test_clean = train_test_split(
    X_noisy_full, X_clean_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

del X_noisy_full, X_clean_full
gc.collect()
print("Test data re-loaded successfully.")


# --- Step 3: Predict using the LOADED Model ---
print("\nPredicting on test data using the saved model...")
decoded_spectra_from_saved_model = saved_autoencoder.predict(X_test_noisy, batch_size=BATCH_SIZE)


# --- Step 4: Visualize a few reconstructions ---
print("\nVisualizing sample reconstructions from the saved model...")
num_samples = 5
indices = np.random.choice(len(X_test_noisy), num_samples, replace=False)

for idx in indices:
    plt.figure(figsize=(12, 5))
    plt.plot(waves, X_test_clean[idx].flatten(), label="Original Clean Spectrum", color='blue')
    plt.plot(waves, X_test_noisy[idx].flatten(), label="Noisy Input Spectrum", color='gray', alpha=0.6)
    plt.plot(
        waves,
        decoded_spectra_from_saved_model[idx].flatten(),
        label="Denoised (Reconstructed) Spectrum",
        linestyle="--",
        color='red'
    )
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Intensity")
    plt.title(f"Spectrum Reconstruction (from saved model) - Sample {idx}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# --- Step 5: Calculate Final Performance Metrics ---
print("\nCalculating final performance metrics from the saved model...")
mae = mean_absolute_error(X_test_clean, decoded_spectra_from_saved_model)
print(f"Final Mean Absolute Error (MAE): {mae:.6f}")

mse = mean_squared_error(X_test_clean, decoded_spectra_from_saved_model)
print(f"Final Mean Squared Error (MSE): {mse:.6f}")

# Flatten for R² score
r2 = r2_score(X_test_clean.flatten(), decoded_spectra_from_saved_model.flatten())
print(f"Final Coefficient of Determination (R²): {r2:.6f}")
