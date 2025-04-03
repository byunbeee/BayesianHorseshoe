import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

def plot_trace(trace, var_names=None):
    """
    Plot trace plots for the given MCMC trace using ArviZ.

    Parameters
    ----------
    trace : pymc3.backends.base.MultiTrace
        The trace object containing MCMC samples.
    var_names : list of str or None, default=None
        List of variable names to include in the trace plot. If None, plots all variables.
    """
    az.plot_trace(trace, var_names=var_names)
    plt.tight_layout()
    plt.show()

def effective_sample_size(trace, var_names=None):
    """
    Calculate and return the effective sample size for variables in the trace.

    Parameters
    ----------
    trace : pymc3.backends.base.MultiTrace
        The trace object containing MCMC samples.
    var_names : list of str or None, default=None
        List of variable names for which to compute the effective sample size. 
        If None, computes for all variables.

    Returns
    -------
    dict
        A dictionary mapping variable names to their effective sample sizes.
    """
    summary_df = pm.summary(trace, var_names=var_names)
    # 'n_eff' column contains the effective sample sizes.
    return summary_df['n_eff'].to_dict()

if __name__ == "__main__":
    # Demo: Generate synthetic data, build the model, run inference, and perform diagnostics.
    from .data import simulate_data
    from .model import build_horseshoe_model
    from .inference import run_mcmc

    # Generate synthetic data
    X, y, true_beta = simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=42)
    
    # Build the Bayesian Horseshoe model
    model = build_horseshoe_model(X, y)
    
    # Run MCMC sampling
    trace = run_mcmc(model)
    
    # Plot trace for the regression coefficients (beta)
    print("Displaying trace plots for 'beta' parameters:")
    plot_trace(trace, var_names=["beta"])
    
    # Compute and print effective sample sizes for beta
    ess = effective_sample_size(trace, var_names=["beta"])
    print("Effective sample sizes for 'beta':")
    for param, neff in ess.items():
        print(f"{param}: {neff}")
