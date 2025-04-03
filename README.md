# Bayesian Variable Selection with Horseshoe Priors

Bayesian Variable Selection with Horseshoe Priors is a Python package designed to implement Bayesian linear regression using Horseshoe Priors for variable selection in high-dimensional datasets. The package provides tools for data simulation, model specification, MCMC-based inference, and diagnostic visualizations.

## Overview

In many regression problems, especially with high-dimensional data, only a few predictors are truly relevant. The Horseshoe Prior offers a powerful Bayesian approach to shrink the coefficients of irrelevant variables toward zero while preserving the signal in the important ones. This package demonstrates how to:

- **Simulate Data:** Generate synthetic regression datasets with sparse true coefficients.
- **Build a Model:** Construct a Bayesian linear regression model with Horseshoe Priors using PyMC3.
- **Run Inference:** Use MCMC sampling to obtain posterior distributions of model parameters.
- **Diagnose and Visualize:** Evaluate the model with trace plots and compute effective sample sizes.

## Features

- **Data Simulation:** Create synthetic datasets with configurable sample size, number of predictors, and sparsity.
- **Model Specification:** Define a Bayesian model with a Horseshoe Prior to encourage sparsity.
- **Inference Engine:** Perform MCMC sampling using the No-U-Turn Sampler (NUTS) implemented in PyMC3.
- **Diagnostics:** Generate trace plots and effective sample size metrics to assess model convergence.
- **Extensible Design:** A modular structure that allows easy integration of additional Bayesian methods or diagnostics.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/yourusername/BayesianHorseshoe.git
cd BayesianHorseshoe
pip install -e .

Usage
Below is a simple example demonstrating the core workflow:

python
Copy
from bayesian_horseshoe.data import simulate_data
from bayesian_horseshoe.model import build_horseshoe_model
from bayesian_horseshoe.inference import run_mcmc
from bayesian_horseshoe.diagnostics import plot_trace, effective_sample_size
import pymc3 as pm

# Simulate synthetic data
X, y, true_beta = simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=42)

# Build the Bayesian model with Horseshoe Prior
model = build_horseshoe_model(X, y)

# Run MCMC sampling
trace = run_mcmc(model, draws=2000, tune=1000, target_accept=0.9, cores=1)

# Summarize the posterior samples for the regression coefficients
print(pm.summary(trace, var_names=["beta"]))

# Visualize trace plots for the regression coefficients
plot_trace(trace, var_names=["beta"])

# Compute and display effective sample sizes for the 'beta' parameters
ess = effective_sample_size(trace, var_names=["beta"])
print("Effective sample sizes for 'beta':")
for param, neff in ess.items():
    print(f"{param}: {neff}")
Notebook Demonstration
A detailed Jupyter Notebook (horseshoe_example.ipynb) is provided in the examples/ directory. This notebook walks through the full process—from data simulation to inference and diagnostics—with interactive visualizations.
