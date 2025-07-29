# %% [markdown]
# 

# %%
import multirex as mrex
import matplotlib.pyplot as plt

# import seaborn as sns
import numpy as np

# import sys
import pandas as pd
import os
import re
import gc
import warnings

# import joblib
import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker


def remove_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Ignore warnings from the custom attributes in pandas
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Pandas doesn't allow columns to be created via a new attribute name*",
    )


from sklearn.model_selection import train_test_split

# Initial setup
remove_warnings()
waves = np.loadtxt("waves.txt")
n_points = len(waves)
indices = np.linspace(0, len(waves) - 1, n_points, endpoint=True)
indices = np.round(indices).astype(int)  # Redondear los índices y convertir a entero

# Seleccionar los elementos de la lista usando los índices
puntos_seleccionados = waves[indices]
waves = puntos_seleccionados
wn_grid = np.sort((10000 / waves))

# %% [markdown]
# ## load data
# 

# %%
def apply_contaminations_from_files(contamination_files, df, n_points):
    """
    Applies multiple contaminations to the data from a list of contamination files
    and returns a DataFrame with all combinations, including the non-contaminated case.

    Parameters:
        contamination_files (list of str): Paths to .txt files containing contaminations.
        df (pandas.DataFrame): Original DataFrame to apply contaminations.
        n_points (int): Number of columns to which the contamination will be applied.

    Returns:
        pandas.DataFrame: DataFrame with all combinations of contaminations, including the
        non-contaminated case, with additional columns 'f_spot' and 'f_fac'.
    """

    # This version is optimized to prevent DataFrame fragmentation.

    df_list = []
    # Non-contaminated case: create a copy and add f_spot and f_fac as 0.0
    df_no_contam = df.copy()
    # Use assign to create new columns efficiently
    df_no_contam = df_no_contam.assign(f_spot=0.0, f_fac=0.0)
    # Reorder
    cols = ["f_spot", "f_fac"] + [
        col for col in df.columns if col not in ["f_spot", "f_fac"]
    ]
    df_no_contam = df_no_contam[cols]
    df_list.append(df_no_contam)

    pattern = r"fspot(?P<f_spot>[0-9.]+)_ffac(?P<f_fac>[0-9.]+)\.txt$"

    for file_path in contamination_files:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        filename = os.path.basename(file_path)
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(
                f"The file name '{filename}' does not match the expected pattern."
            )

        f_spot = float(match.group("f_spot"))
        f_fac = float(match.group("f_fac"))

        try:
            contamination_data = np.loadtxt(file_path, ndmin=2)
            contam_values = (
                contamination_data[:, 1]
                if contamination_data.shape[1] >= 2
                else contamination_data.flatten()
            )
            if len(contam_values) != n_points:
                raise ValueError(
                    f"Contamination values in '{filename}' ({len(contam_values)}) != n_points ({n_points})."
                )
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")

        contam_values = contam_values[::-1]

        df_contam = df.copy()
        data_columns = df_contam.columns[-n_points:]

        # Perform multiplication
        df_contam[data_columns] = df_contam[data_columns].multiply(
            contam_values, axis=1
        )

        # Create new columns efficiently using assign
        df_contam = df_contam.assign(f_spot=f_spot, f_fac=f_fac)

        # Reorder columns
        cols = ["f_spot", "f_fac"] + [
            col for col in df.columns if col not in ["f_spot", "f_fac"]
        ]
        df_contam = df_contam[cols]
        df_list.append(df_contam)

    df_final = pd.concat(
        df_list, ignore_index=True
    ).copy()  # .copy() ensures a de-fragmented frame
    df_final.data = df_final.iloc[:, -n_points:]
    df_final.params = df_final.iloc[:, :-n_points]
    return df_final

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


def filter_rows(df):
    """
    Filters rows of a DataFrame where at least one of the columns
    "atm CH4", "atm O3", or "atm H2O" has a value >= -8.
    Returns the DataFrame unchanged if none of these columns are present.
    """
    filter_columns = ["atm CH4", "atm O3", "atm H2O"]
    present_columns = [col for col in filter_columns if col in df.columns]

    for chem in present_columns:
        df = df[df[chem] >= -8]
        # Set .data and .params attributes on the final DataFrame
    df.data = df.iloc[:, -n_points:]
    df.params = df.iloc[:, :-n_points]
    return df


# Helper function to load data and correctly set dtypes
def load_and_prep_data(filepath, n_points):
    """Reads a CSV and converts only the spectral data columns to float32."""
    df = pd.read_csv(filepath)
    # Identify spectral data columns (last n_points)
    data_cols = df.columns[-n_points:]
    # Convert only spectral data to float32, leave params as they are
    df[data_cols] = df[data_cols].astype("float32")
    return df


try:
    airless_data = load_and_prep_data("spec_data/airless_data.csv", n_points)
    airless_data = apply_contaminations_from_files(
        contamination_files, airless_data, n_points
    )

    CO2_data = load_and_prep_data("spec_data/CO2_data.csv", n_points)
    CO2_data = apply_contaminations_from_files(contamination_files, CO2_data, n_points)

    CH4_data = load_and_prep_data("spec_data/CH4_data.csv", n_points)
    CH4_data = filter_rows(CH4_data)
    CH4_data = apply_contaminations_from_files(contamination_files, CH4_data, n_points)
except Exception as e:
    print(f"Error processing initial data: {e}")

# %% [markdown]
# ## Clean data
# 

# %%
def mult_df(df, n_points, n_mult):
    df_list = []
    for _ in range(n_mult + 1):
        df_no_contam = df.copy()
        # Use assign to avoid fragmentation
        df_no_contam = df_no_contam.assign(f_spot=0.0, f_fac=0.0)
        cols = ["f_spot", "f_fac"] + [
            col for col in df.columns if col not in ["f_spot", "f_fac"]
        ]
        df_no_contam = df_no_contam[cols]
        df_list.append(df_no_contam)

    df_final = pd.concat(df_list, ignore_index=True).copy()  # .copy() de-fragments
    df_final.data = df_final.iloc[:, -n_points:]
    df_final.params = df_final.iloc[:, :-n_points]
    return df_final

# %%
try:
    airless_data_clean = load_and_prep_data("spec_data/airless_data.csv", n_points)
    airless_data_clean = mult_df(airless_data_clean, n_points, 9)

    CO2_data_clean = load_and_prep_data("spec_data/CO2_data.csv", n_points)
    CO2_data_clean = mult_df(CO2_data_clean, n_points, 9)

    CH4_data_clean = load_and_prep_data("spec_data/CH4_data.csv", n_points)
    CH4_data_clean = filter_rows(CH4_data_clean)
    CH4_data_clean = mult_df(CH4_data_clean, n_points, 9)
except Exception as e:
    print(f"Error processing clean data: {e}")

# %%
def normalize_min_max_by_row(df):
    min_by_row = df.min(axis=1)
    max_by_row = df.max(axis=1)
    range_by_row = max_by_row - min_by_row
    # Avoid division by zero
    range_by_row[range_by_row == 0] = 1
    normalized = (df.sub(min_by_row, axis=0)).div(range_by_row, axis=0)
    return normalized

# %% [markdown]
# # Noisy and Clean
# 

# %%
def normalize_min_max_by_row(df):
    min_by_row = df.min(axis=1)
    max_by_row = df.max(axis=1)
    range_by_row = max_by_row - min_by_row
    # Avoid division by zero
    range_by_row[range_by_row == 0] = 1
    normalized = (df.sub(min_by_row, axis=0)).div(range_by_row, axis=0)
    return normalized


def generate_df_with_noise_std(df, n_repeat, noise_std, seed=None):
    if not hasattr(df, "params"):
        df_params = pd.DataFrame()
        if not hasattr(df, "data"):
            df_spectra = df
    else:
        if not hasattr(df, "data"):
            raise ValueError("The DataFrame must have a 'data' attribute.")
        df_params = df.params
        df_spectra = df.data

    df_spectra = df_spectra.astype("float32")

    if seed is not None:
        np.random.seed(seed)

    df_spectra_replicated = pd.DataFrame(
        np.repeat(df_spectra.values, n_repeat, axis=0),
        columns=df_spectra.columns,
    )

    if isinstance(noise_std, (int, float)):
        noise_replicated = np.full(
            df_spectra_replicated.shape, noise_std, dtype="float32"
        )
    else:
        noise_array = np.array(noise_std, dtype="float32")
        noise_replicated = np.repeat(noise_array[:, np.newaxis], n_repeat, axis=0)
        noise_replicated = np.tile(
            noise_replicated, (1, df_spectra_replicated.shape[1])
        )

    gaussian_noise = np.random.normal(
        0, noise_replicated, df_spectra_replicated.shape
    ).astype("float32")
    df_spectra_replicated += gaussian_noise

    df_params_replicated = pd.DataFrame(
        np.repeat(df_params.values, n_repeat, axis=0),
        columns=df_params.columns,
    )

    # Efficiently add new columns
    new_cols_data = {
        "noise_std": (
            np.repeat(noise_std, n_repeat)
            if isinstance(noise_std, (list, np.ndarray, pd.Series))
            else noise_std
        ),
        "n_repeat": n_repeat,
    }
    new_cols_df = pd.DataFrame(new_cols_data, index=df_params_replicated.index)

    # Concatenate all parts at once
    df_final = pd.concat(
        [
            new_cols_df,
            df_params_replicated.reset_index(drop=True),
            df_spectra_replicated.reset_index(drop=True),
        ],
        axis=1,
    )

    df_final.data = df_final.iloc[:, -df_spectra_replicated.shape[1] :]
    df_final.params = df_final.iloc[:, : -df_spectra_replicated.shape[1]]

    return df_final

# %% [markdown]
# ### OPTIMIZED: Generate and Process All Data
# 
# This single cell replaces all the individual SNR cells and the final `## Data` cell.
# It loops through each SNR value, generates the noisy and clean data, normalizes it,
# appends the result to a list as a NumPy array, and then clears the memory.
# This avoids holding multiple massive dataframes in memory at once.
# 

import os

# Define file paths for the cached numpy arrays in a dedicated directory
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
noisy_path = os.path.join(output_dir, "X_noisy_full_dataset.npy")
clean_path = os.path.join(output_dir, "X_clean_full_dataset.npy")

# Caching Logic: Check if files exist
if os.path.exists(noisy_path) and os.path.exists(clean_path):
    print("Found cached data. Loading from disk...")
    X_noisy = np.load(noisy_path)
    X_no_noisy = np.load(clean_path)
    print("...Loading complete.")
    print(f"Loaded noisy data shape: {X_noisy.shape}")
    print(f"Loaded clean data shape: {X_no_noisy.shape}")

else:
    print("Cached data not found. Starting data generation process...")

    list_of_noisy_arrays = []
    list_of_clean_arrays = []

    # Using None for the 'Nan' (no noise) case
    snr_values = [1, 3, 6, 10, None]

    for snr in snr_values:
        print(
            f"--- Processing SNR = {snr if snr is not None else 'inf (no additional noise)'} ---"
        )
        gc.collect()

        # --- Generate Noisy Data (Contaminated + Noise) ---
        if snr is not None:
            noise = mrex.generate_df_SNR_noise(df=CO2_data, n_repeat=1, SNR=snr)[
                "noise"
            ][0]
        else:
            noise = 0.0  # No additional noise for the "Nan" case

        temp_CO2_data = generate_df_with_noise_std(
            df=CO2_data, n_repeat=5000, noise_std=noise
        )
        temp_CH4_data = generate_df_with_noise_std(
            df=CH4_data, n_repeat=500, noise_std=noise
        )
        temp_airless_data = generate_df_with_noise_std(
            df=airless_data, n_repeat=5000, noise_std=noise
        )

        current_noisy_df = pd.concat(
            [temp_CO2_data, temp_CH4_data, temp_airless_data], ignore_index=True
        )

        normalized_data = normalize_min_max_by_row(current_noisy_df.iloc[:, -n_points:])
        list_of_noisy_arrays.append(normalized_data.values.astype("float32"))

        del (
            temp_CO2_data,
            temp_CH4_data,
            temp_airless_data,
            current_noisy_df,
            normalized_data,
        )
        gc.collect()

        # --- Generate Corresponding Clean Data (Uncontaminated + No Noise) ---
        temp_no_CO2_data = generate_df_with_noise_std(
            df=CO2_data_clean, n_repeat=5000, noise_std=0
        )
        temp_no_CH4_data = generate_df_with_noise_std(
            df=CH4_data_clean, n_repeat=500, noise_std=0
        )
        temp_no_airless_data = generate_df_with_noise_std(
            df=airless_data_clean, n_repeat=5000, noise_std=0
        )

        current_clean_df = pd.concat(
            [temp_no_CO2_data, temp_no_CH4_data, temp_no_airless_data],
            ignore_index=True,
        )

        normalized_data = normalize_min_max_by_row(current_clean_df.iloc[:, -n_points:])
        list_of_clean_arrays.append(normalized_data.values.astype("float32"))

        del (
            temp_no_CO2_data,
            temp_no_CH4_data,
            temp_no_airless_data,
            current_clean_df,
            normalized_data,
        )
        gc.collect()

    print("\n--- Final Data Concatenation ---")
    X_noisy = np.concatenate(list_of_noisy_arrays, axis=0)
    del list_of_noisy_arrays

    X_no_noisy = np.concatenate(list_of_clean_arrays, axis=0)
    del list_of_clean_arrays

    gc.collect()

    print("\n--- Saving generated data to disk for future runs ---")
    np.save(noisy_path, X_noisy)
    np.save(clean_path, X_no_noisy)
    print(f"Data saved to directory: '{output_dir}'")

    print(f"\nFinal noisy data shape: {X_noisy.shape}")
    print(f"Final clean data shape: {X_no_noisy.shape}")
    assert (
        X_noisy.shape[0] == X_no_noisy.shape[0]
    ), "The number of samples does not match."

# %% [markdown]
# ### FInal data
# 

# %% [markdown]
# ### Final data prep and Model Training
# 
# The rest of your script remains the same.
# 

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# %%
test_size = 0.2

X_train_noisy, X_test_noisy, X_train_clean, X_test_clean = train_test_split(
    X_noisy, X_no_noisy, test_size=test_size, random_state=42
)
del (X_noisy, X_no_noisy)
gc.collect()

# %% [markdown]
# ### Dense Autoencoder
# 

# %%
input_dim = X_train_noisy.shape[1]

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
encoded = layers.Dropout(0.2)(encoded)

# Decoder
decoded = layers.Dense(300, activation="swish")(encoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(300, activation="swish")(decoded)
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

# %%


# %%
# Train the autoencoder
history = autoencoder.fit(
    X_train_noisy,
    X_train_clean,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test_noisy, X_test_clean),
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ],
)

# Save the trained model
autoencoder.save("AE_CH4.keras")

# Plot training and validation MAE
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training MAE")
plt.plot(history.history["val_loss"], label="Validation MAE")
plt.title("MAE Progress During Training")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

# Predict reconstructed spectra on test data
decoded_spectra = autoencoder.predict(X_test_noisy)

# Visualize a few reconstructions
num_samples = 5  # Number of samples to visualize
indices = np.random.choice(len(X_test_noisy), num_samples, replace=False)

for idx in indices:
    plt.figure(figsize=(10, 4))
    plt.plot(waves, X_test_clean[idx].flatten(), label="Original Clean Spectrum")
    plt.plot(
        waves, X_test_noisy[idx].flatten(), label="Noisy Input Spectrum", alpha=0.5
    )
    plt.plot(
        waves,
        decoded_spectra[idx].flatten(),
        label="Denoised (Reconstructed) Spectrum",
        linestyle="--",
    )
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Intensity")
    plt.title(f"Spectrum Reconstruction - Sample {idx}")
    plt.legend()
    plt.show()

# %% [markdown]
# # Eval
# 

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Re-load the best model saved by EarlyStopping
autoencoder = keras.models.load_model("AE_CH4.keras")
X_reconstructed = autoencoder.predict(X_test_noisy)

# Compute evaluation metrics on the test set
mae = mean_absolute_error(X_test_clean, X_reconstructed)
print(f"Mean Absolute Error (MAE): {mae:.6f}")

mse = mean_squared_error(X_test_clean, X_reconstructed)
print(f"Mean Squared Error (MSE): {mse:.6f}")

r2 = r2_score(X_test_clean.flatten(), X_reconstructed.flatten())
print(f"Coefficient of Determination (R²): {r2:.6f}")


