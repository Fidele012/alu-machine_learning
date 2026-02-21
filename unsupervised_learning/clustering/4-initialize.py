#!/usr/bin/env python3
"""
Gaussian Mixture Model
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''
    A function that initializes cluster centroids for
    a GMM
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    C, clss = kmeans(X, k)
    pi = np.full(k, 1 / k)
    m = C
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
