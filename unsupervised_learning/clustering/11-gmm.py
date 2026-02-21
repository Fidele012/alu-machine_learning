#!/usr/bin/env python3
'''
Calculates the Gaussian Matrix
'''


import sklearn.mixture


def gmm(X, k):
    """
    Calculates a Gaussian Mixture Model (GMM) from a dataset.

    """
    # Fit the Gaussian Mixture Model to the data
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    # Extract the required parameters
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
