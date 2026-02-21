#!/usr/bin/env python3
"""
Defines function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""


import numpy as np
expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC
    """
    return None, None, None, None
