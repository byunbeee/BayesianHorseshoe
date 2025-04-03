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


