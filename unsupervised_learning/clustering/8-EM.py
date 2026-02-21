#!/usr/bin/env python3
'''
Exoectation Maximization for a Gaussian Mixture Model
'''


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    '''
    Expectation maximization for a GMM
    '''
    if not isinstance(X, np.ndarray):
        return None, None, None, None, None
    if not isinstance(k, int):
        return None, None, None, None, None
    if not isinstance(iterations, int):
        return None, None, None, None, None
    if not isinstance(tol, float):
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, l = expectation(X, pi, m, S)
    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, l = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")
    return pi, m, S, g, l
