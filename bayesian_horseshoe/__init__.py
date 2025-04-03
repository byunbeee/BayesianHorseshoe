"""
Bayesian Variable Selection with Horseshoe Priors Package

This package implements Bayesian variable selection using Horseshoe Priors,
providing modules for data simulation, model specification, inference, and diagnostics.

Modules:
- bayesian_horseshoe.data: Functions for data simulation and preprocessing.
- bayesian_horseshoe.model: Bayesian model definition using Horseshoe Priors.
- bayesian_horseshoe.inference: Inference routines (e.g., MCMC or variational inference).
- bayesian_horseshoe.diagnostics: Tools for convergence diagnostics and model evaluation.
"""

__version__ = "0.1.0"
__author__ = "Hyun Wook Sim"  

from . import data
from . import model
from . import inference
from . import diagnostics

__all__ = ["data", "model", "inference", "diagnostics"]
