# **Lumos: AI-Driven Exoplanet Biosignature Classification**

**Lumos is a dual-component project designed to accelerate the search for life in the universe. This repository contains the AI Analysis Pipeline used to detect potential biosignatures in noisy exoplanet data. The [Educational Game Interface](https://github.com/bunrots/Lumos) is in a separate repository.**

## **Table of Contents**

- [About The Project](#about-the-project)
  - [The Challenge](#the-challenge)
  - [Our Solution](#our-solution)
- [Key Results & Highlights](#key-results--highlights)
- [Built With](#built-with)
- [Methodology: The AI Pipeline](#methodology-the-ai-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## **About The Project**

### **The Challenge**

Modern instruments like the James Webb Space Telescope (JWST) can analyze the atmospheres of distant exoplanets. However, the signals (spectra) are incredibly faint and often buried in instrumental and stellar noise. This results in a very low Signal-to-Noise Ratio (SNR).  
Traditional analysis methods require hundreds of hours of telescope time to confirm a signal for a single planet, making the search for life a slow, expensive, and impractical process.  
*(A visual comparison of a clean, theoretical signal vs. a realistic, noisy signal our AI is trained to analyze)*

### **Our Solution**

Lumos addresses this challenge with a two-part system:

1. **(This Repo) AI Analysis Pipeline**: An end-to-end machine learning workflow that automatically processes, cleans, and classifies noisy exoplanet spectra to identify promising candidates for further study.  
2. [**Educational Game Interface**](https://github.com/bunrots/Lumos): An interactive 3D application developed in Unity that visualizes the AI's findings, allowing anyone to explore the TRAPPIST-1 system and understand the results in an engaging way.

## **Key Results & Highlights**

Our AI pipeline demonstrated high accuracy and efficiency in identifying potential biosignatures (e.g., CH₄, H₂O, O₃):

* ⭐ **98.8% Variance Explained**: Our Denoising Autoencoder (built with TensorFlow/Keras) successfully reconstructed clean spectra from noisy inputs, achieving an R² score of 0.988.  
* 🎯 **Up to 97% F1-Score**: The final XGBoost classifier achieved a 96% Recall and 97% F1-Score in identifying biosignatures, minimizing the chance of missing a potentially habitable world.  
* 🚀 **3.5x Faster Training**: By leveraging GPU acceleration, our XGBoost model trained approximately **3.5 times faster** than the baseline Random Forest model, proving its efficiency for larger datasets.

## **Built With**

This project was brought to life using a combination of powerful technologies for data science and game development.  
**AI & Data Science (This Repository):**

* [Python 3.11](https://www.python.org/)  
* [TensorFlow / Keras](https://www.tensorflow.org/) (for Denoising Autoencoder)  
* [Scikit-learn](https://scikit-learn.org/) (for Random Forest & evaluation)  
* [XGBoost](https://xgboost.ai/) (for the high-performance classifier)  
* [Pandas](https://pandas.pydata.org/)  
* [NumPy](https://numpy.org/)  
* [Jupyter Notebooks](https://jupyter.org/) (for development and experimentation)  
* [Matplotlib / Seaborn](https://matplotlib.org/) (for visualization)

**Game & Visualization ([bunrots/Lumos](https://github.com/bunrots/Lumos)):**

* [Unity Engine](https://unity.com/)  
* [C\#](https://docs.microsoft.com/en-us/dotnet/csharp/)

## **Methodology: The AI Pipeline**

The core of this repository is a multi-stage AI workflow inspired by the latest academic research.

1. **Synthetic Data Generation**: We created a large dataset of over 700,000 synthetic exoplanet spectra (based on TRAPPIST-1e), incorporating various atmospheric compositions, stellar contamination, and noise levels (SNR 1-10).  
2. **Denoising with Autoencoder**: A deep learning autoencoder was trained to "clean" the noisy spectra, isolating the underlying atmospheric signal from the noise.  
3. **Model Training**: The cleaned spectra were used as input to train two separate classifiers for comparison: a standard Random Forest and a GPU-accelerated XGBoost model.  
4. **Evaluation & Export**: We evaluated the models using F1-Score, Precision, and Recall. The final classification results were exported as a JSON file (planetdata.json) to be consumed by the Unity game.

## **Getting Started**

To get a local copy up and running, follow these simple steps.

### **Prerequisites**

You will need the following software installed on your machine:

* Python 3.11 or later  
* Pip (Python package installer)  
* Git

### **Installation**

1. **Clone this (the AI) repository:**  
   git clone \[https://github.com/chikobara/Lumos.git\](https://github.com/chikobara/Lumos.git)  
   cd Lumos

2. **Set up a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

3. **Install Python dependencies:**  
   pip install \-r requirements.txt

## **Usage**

The AI pipeline is organized into a series of Jupyter Notebooks that should be run in sequence. Open this directory in a code editor that supports notebooks, such as VS Code or Jupyter Lab.

1. 01\_... (Optional) Notebooks for initial data exploration.  
2. 02\_Data\_Generation.ipynb: Run this notebook first to generate the full noisy and clean datasets.  
   * *Note: This is computationally intensive and may take a long time.*  
3. 03\_AE\_...ipynb: Run the appropriate notebook (e.g., 03\_AE\_CH4.ipynb) to train the Denoising Autoencoder for a specific biosignature. This will save a .keras model file.  
4. 04\_...\_RF.ipynb: Run the corresponding notebook (e.g., 04\_CH4\_RF.ipynb) to load the trained autoencoder, clean the data, and then train and evaluate the Random Forest and XGBoost classifiers.

The final JSON output (planetdata.json) used by the game is generated from these notebooks.  
For instructions on running the game interface, please see the [**Lumos Game Repository (bunrots/Lumos)**](https://github.com/bunrots/Lumos).
