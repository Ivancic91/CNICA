import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Generate correlated Gaussian data
# ------------------------------------------------------------
def generate_correlated_data(N=2, T=5000, rho=0.6, seed=0):
    """
    N   : dimension
    T   : number of samples
    rho : pairwise correlation
    """
    rng = np.random.default_rng(seed)
    Sigma = np.ones((N, N)) * rho
    np.fill_diagonal(Sigma, 1.0)

    # Cholesky factor
    L = np.linalg.cholesky(Sigma)

    # Standard normal samples
    Z = rng.standard_normal((N, T))

    # Correlated samples
    X = L @ Z
    return X, Sigma

# ------------------------------------------------------------
# Exact objective:
#   -log(det(Sigma) / prod(diag(Sigma)))
# ------------------------------------------------------------
def exact_objective(Sigma):
    diag = np.diag(Sigma)
    return -np.log(np.linalg.det(Sigma) / np.prod(diag))

# ------------------------------------------------------------
# Show example
# ------------------------------------------------------------
def plot_example(approx_objective, N=2, T=5000, rho=0.6):
    """
    N   : dimension
    T   : number of samples
    rho : pairwise correlation
    """

    # Generate data
    X, Sigma_true = generate_correlated_data(N=N, T=T, rho=rho)

    # Sample covariance
    Sigma_hat = (X @ X.T) / T

    # Compute objectives
    exact_val = exact_objective(Sigma_hat)
    approx_val = approx_objective(Sigma_hat)

    # Print results
    print("Sample covariance:")
    print(Sigma_hat)
    print()
    print("Exact objective:")
    print("  -log(det(Sigma) / prod(diag)) =", exact_val)
    print()
    print("Quadratic approximation:")
    print("  sum rho_ij^2 =", approx_val)

    # Plot
    plt.figure()
    if N == 2:
        plt.scatter(X[0], X[1], s=3, alpha=0.5)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("2D Correlated Gaussian Samples")
    else:
        # For 3D, plot first two coordinates
        plt.scatter(X[0], X[1], s=3, alpha=0.5)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"{N}D Correlated Gaussian (X1 vs X2)")

    plt.axis("equal")
    plt.grid(True)
    plt.show()