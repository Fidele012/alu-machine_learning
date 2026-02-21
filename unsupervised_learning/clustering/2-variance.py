#!/usr/bin/env python3
"""
A function def variance(X, C): that calculates
the total intra-cluster variance for a data set
"""


import numpy as np


def variance(X, C):
    """
        A function def variance(X, C): that calculates
        the total intra-cluster variance for a data set

        Args:
        X is a numpy.ndarray of shape (n, d)
        containing the data set
        C is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster

        Returns:
        - var, or None on failure
        - var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray) or \
            len(X.shape) != 2 or len(C.shape) != 2 or \
            X.shape[1] != C.shape[1] or C.shape[1] <= 0 or X.size == 0 or \
            C.size == 0:
        return None

    dist_diff = np.linalg.norm(X - C[:, np.newaxis], axis=2).T
    minimum_dist = np.min(dist_diff, axis=1)
    var = np.sum(np.square(minimum_dist))
    return var
