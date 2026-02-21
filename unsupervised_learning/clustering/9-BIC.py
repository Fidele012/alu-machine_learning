#!/usr/bin/env python3
"""
Defines function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
        kmax: positive integer containing the maximum number of clusters
        iterations: positive integer containing max iterations for EM
        tol: non-negative float containing tolerance for EM
        verbose: boolean that determines if EM should print info
    
    Returns:
        best_k: best value for k based on its BIC
        best_result: tuple containing (pi, m, S) for best k
        l: numpy.ndarray of shape (kmax - kmin + 1) containing log likelihood
        b: numpy.ndarray of shape (kmax - kmin + 1) containing BIC values
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    
    if type(kmin) != int or kmin < 1:
        return None, None, None, None
    
    n, d = X.shape
    
    # Set kmax to maximum possible clusters if not specified
    if kmax is None:
        kmax = n
    
    if type(kmax) != int or kmax < 1:
        return None, None, None, None
    
    if kmin >= kmax:
        return None, None, None, None
    
    if type(iterations) != int or iterations < 1:
        return None, None, None, None
    
    if tol < 0:
        return None, None, None, None
    
    # Initialize arrays to store results
    num_tests = kmax - kmin + 1
    l = np.zeros(num_tests)  # Log likelihoods
    b = np.zeros(num_tests)  # BIC values
    results = []  # Store all results
    
    # Test each cluster size from kmin to kmax
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
        
        # Calculate number of parameters
        # For k Gaussian components:
        # - pi: k-1 parameters (probabilities sum to 1)
        # - means: k * d parameters
        # - covariances: k * d * (d + 1) / 2 parameters (symmetric matrices)
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        # Alternative formulation: p = k * (1 + d + d * (d + 1) / 2) - 1
        
        # Calculate BIC: BIC = p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * log_likelihood
    
    # Find the best k (minimum BIC)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]
    
    return best_k, best_result, l, b
