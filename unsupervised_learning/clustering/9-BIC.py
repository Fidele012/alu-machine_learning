#!/usr/bin/env python3
"""
Defines function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
              to check for (inclusive)
        kmax: positive integer containing the maximum number of clusters
              to check for (inclusive). If None, set to maximum possible
        iterations: positive integer containing the maximum number of
                    iterations for the EM algorithm
        tol: non-negative float containing the tolerance for the EM algorithm
        verbose: boolean that determines if the EM algorithm should print
                 information to the standard output

    Returns:
        best_k: best value for k based on its BIC
        best_result: tuple containing (pi, m, S)
            pi: numpy.ndarray of shape (k,) containing the cluster priors
                for the best number of clusters
            m: numpy.ndarray of shape (k, d) containing the centroid means
               for the best number of clusters
            S: numpy.ndarray of shape (k, d, d) containing the covariance
               matrices for the best number of clusters
        l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
           likelihood for each cluster size tested
        b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
           value for each cluster size tested
           BIC = p * ln(n) - 2 * l
           where p is the number of parameters, n is the number of data
           points, and l is the log likelihood

    Returns None, None, None, None on failure
    """
    # Validate X
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    # Validate kmin
    if not isinstance(kmin, int) or kmin < 1 or kmin > n:
        return None, None, None, None

    # Set kmax to n if None
    if kmax is None:
        kmax = n

    # Validate kmax
    if not isinstance(kmax, int) or kmax < 1 or kmax > n:
        return None, None, None, None

    # Validate kmin <= kmax
    if kmin > kmax:
        return None, None, None, None

    # Validate iterations
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None

    # Validate tol
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    # Validate verbose
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Initialize arrays to store results
    num_tests = kmax - kmin + 1
    l = np.zeros(num_tests)
    b = np.zeros(num_tests)
    results = []

    # Test each cluster size from kmin to kmax (single loop)
    for k in range(kmin, kmax + 1):
        idx = k - kmin

        # Run expectation maximization for current k
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )

        # Check if EM failed
        if pi is None:
            return None, None, None, None

        # Store results
        results.append((pi, m, S))
        l[idx] = log_likelihood

        # Calculate number of parameters for GMM with k components:
        # - pi: k - 1 free parameters (probabilities must sum to 1)
        # - means (m): k * d parameters
        # - covariances (S): k * d * (d + 1) / 2 parameters
        #   (each covariance matrix is symmetric, so d*(d+1)/2 unique values)
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        # Calculate BIC: BIC = p * ln(n) - 2 * l
        # Lower BIC indicates better model (trade-off between fit and
        # complexity)
        b[idx] = p * np.log(n) - 2 * log_likelihood

    # Find the best k (minimum BIC value)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
