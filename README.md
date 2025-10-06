# MultiREx - Planetary Transmission Spectra Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D4san/MultiREx-public/blob/main/examples/multirex-quickstart.ipynb)

MultiREx is a Python package designed to generate planetary transmission spectra. It is a powerful tool for researchers and enthusiasts in the field of astrophysics, enabling the creation of synthetic spectra for Earth-like exoplanets. The package is particularly useful for generating large datasets for machine learning applications, as demonstrated in the research paper "Machine-assisted classification of potential biosignatures in earth-like exoplanets using low signal-to-noise ratio transmission spectra" (arXiv:2407.19167).

## Installation

You can install MultiREx using pip:

```bash
pip install multirex
```

### Dependencies

MultiREx relies on the following external packages:

- `pandexo.engine`
- `plotly`
- `scikit-learn`
- `mpi4py`
- `taurex`

These dependencies will be installed automatically when you install MultiREx via pip.

## Quick Start

Here's a quick example of how to use MultiREx to create a system and generate a transmission spectrum:

```python
import multirex as mrex
import numpy as np
import matplotlib.pyplot as plt

# Create a system
system = mrex.System(
    star=mrex.Star(temperature=5777, radius=1, mass=1),
    planet=mrex.Planet(
        radius=1,
        mass=1,
        atmosphere=mrex.Atmosphere(
            temperature=290,  # in K
            base_pressure=1e5,  # in Pa
            top_pressure=1,  # in Pa
            fill_gas="N2",  # the gas that fills the atmosphere
            composition=dict(
                CO2=-4,  # This is the log10(mix-ratio)
                CH4=-6,
            ),
        ),
    ),
    sma=1,
)

# Generate the transmission spectrum
system.make_tm()
wns = mrex.Physics.wavenumber_grid(wl_min=0.6, wl_max=10, resolution=300)
wns, spectrum = system.generate_spectrum(wns)
wls = 1e4 / wns

# Plot the spectrum
plt.plot(wls, spectrum * 1e6)
plt.grid()
plt.xlabel("Wavelength (micron)")
plt.ylabel("Transit depth [ppm]")
plt.show()
```

## Features

- **Planetary System Modeling:** Create detailed models of stars, planets, and their atmospheres.
- **Transmission Spectra Generation:** Generate high-quality theoretical transmission spectra.
- **Random System Generation:** Create random realizations of systems to explore a wide range of parameters.
- **Observed Spectra Simulation:** Generate observed spectra with added noise to simulate real-world observations.
- **Data for Machine Learning:** Ideal for generating large datasets for training machine learning models.

## License

This project is licensed under the terms of the LICENSE file.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.