# Bias-Based Causal Discovery

This repository accompanies the paper *Effects of Distributional Biases on Gradient-Based
Causal Discovery in the Bivariate Categorical Case*.

## Requirements:
This repository was developed on Python 3.10. The required packages are:

- `torch`
- `tqdm`
- `numpy`
- `matplotlib`
- `pathlib`
- `gitpython`
- `scipy`
- `pandas`

When creating videos also:
- `ffmpeg`

## Usage

You can configure model parameters in `utils/model.py`. With `run.py` you can run a model, which will automatically create a folder `plots/` if it does not exist and then a folder for the current experiment inside it. It will save various plots and data there.

For performing multiple runs with different parameters, for instance different `epsilon`, you can use `generate_run_data.py` for this.

If you want to investigate distributions, use the notebooks in `notebooks/`.
