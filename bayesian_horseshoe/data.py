import numpy as np

def simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=None):
    """
    Generate synthetic data for a regression problem with a sparse true coefficient vector.

    Parameters
    ----------
    n : int, default=100
        Number of samples.
    p : int, default=50
        Number of predictors.
    n_relevant : int, default=5
        Number of predictors with non-zero coefficients.
    noise_std : float, default=0.5
        Standard deviation of the Gaussian noise added to the response.
    seed : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : numpy.ndarray
        Generated predictor matrix of shape (n, p).
    y : numpy.ndarray
        Generated response vector of shape (n,).
    true_beta : numpy.ndarray
        True regression coefficients of shape (p,).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate the predictor matrix
    X = np.random.randn(n, p)
    
    # Initialize coefficients as zeros
    true_beta = np.zeros(p)
    
    # Randomly choose indices to be relevant (non-zero coefficients)
    relevant_indices = np.random.choice(p, size=n_relevant, replace=False)
    
    # Assign random values to the selected indices
    true_beta[relevant_indices] = np.random.randn(n_relevant)
    
    # Generate response variable with Gaussian noise
    y = X.dot(true_beta) + np.random.randn(n) * noise_std
    
    return X, y, true_beta

if __name__ == "__main__":
    # Demo: Generate and display synthetic data
    X, y, true_beta = simulate_data(n=100, p=50, n_relevant=5, noise_std=0.5, seed=42)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("True coefficients:", true_beta)