#!/usr/bin/env python3
'''
A function def initialize(X, k):
that initializes cluster centroids for K-means
'''


import numpy as np


def initialize(X, k):
    '''
        X is a numpy.ndarray of shape (n, d) that contains
            the dataset
        k is a positive integer containing
            the number of clusters
        Returns: C, a numpy.ndarray of shape (k, d) containing
            the initialized centroids for each cluster
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    # min values of X along each dimension in d
    low = np.min(X, axis=0)
    # max values of X along each dimension in d
    high = np.max(X, axis=0)
    # initialize cluster centroids with multivariate uniform distribution
    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids
