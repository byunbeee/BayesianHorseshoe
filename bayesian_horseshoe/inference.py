import pymc3 as pm

def run_mcmc(model, draws=2000, tune=1000, target_accept=0.9, cores=1):
    """
    Run MCMC sampling on the provided PyMC3 model using the NUTS sampler.

    Parameters
    ----------
    model : pymc3.Model
        A PyMC3 model instance.
    draws : int, default=2000
        Number of MCMC draws.
    tune : int, default=1000
        Number of tuning steps.
    target_accept : float, default=0.9
        The target acceptance probability for the sampler.
    cores : int, default=1
        Number of parallel chains.

    Returns
    -------
    trace : pymc3.backends.base.MultiTrace
        The trace object containing the posterior samples.
    """
    with model:
        trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, cores=cores, return_inferencedata=False)
    return trace

if __name__ == "__main__":
    # Demo: Build the model using simulated data and run MCMC sampling
    from .data import simulate_data
    from .model import build_horseshoe_model

    # Generate synthetic data
    X, y, true_beta = simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=42)
    
    # Build the Horseshoe model
    model = build_horseshoe_model(X, y)
    
    # Run MCMC sampling
    trace = run_mcmc(model)
    
    # Print a summary of the posterior samples
    print("MCMC sampling completed. Posterior summary:")
    print(pm.summary(trace))