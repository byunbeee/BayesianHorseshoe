import pymc3 as pm
import numpy as np

def build_horseshoe_model(X, y):
    """
    Build a Bayesian linear regression model with Horseshoe Prior on the coefficients.

    Parameters
    ----------
    X : numpy.ndarray
        Predictor matrix of shape (n, p).
    y : numpy.ndarray
        Response vector of shape (n,).

    Returns
    -------
    model : pymc3.Model
        A PyMC3 model instance with the horseshoe prior applied to the regression coefficients.
    """
    n, p = X.shape

    with pm.Model() as model:
        # Global shrinkage parameter (tau) controls overall sparsity
        tau = pm.HalfCauchy('tau', beta=1)
        # Local shrinkage parameters (lam) for each coefficient, encouraging sparsity
        lam = pm.HalfCauchy('lam', beta=1, shape=p)
        # Horseshoe prior for the regression coefficients
        beta = pm.Normal('beta', mu=0, sigma=tau * lam, shape=p)
        # Noise standard deviation parameter
        sigma = pm.HalfNormal('sigma', sigma=1)
        # Define the linear model
        mu = pm.math.dot(X, beta)
        # Likelihood: observed data
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    return model

if __name__ == "__main__":
    # Example usage for testing the model function with simulated data
    from .data import simulate_data
    X, y, true_beta = simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=42)
    
    model = build_horseshoe_model(X, y)
    print("Horseshoe model created successfully!")