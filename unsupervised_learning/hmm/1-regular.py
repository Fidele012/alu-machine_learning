#!/usr/bin/env python3
'''
    Function def regular(P): that determines
    the steady state probabilities of a regular
    markov chain
'''


import numpy as np


def regular(P):
    '''
        Determines the steady state probabilities of a regular
        markov chain
    '''
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return None
    if not (P > 0).all():
        return None
    Identity = np.identity(n)
    Q = P - Identity
    e = np.ones((n,))
    Qe = np.c_[Q, e]
    QTQ = np.matmul(Qe, Qe.T)
    QbT = np.ones((n,))
    result = np.linalg.solve(QTQ, QbT)
    return np.expand_dims(result, axis=0)
