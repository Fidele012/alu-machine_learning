#!/usr/bin/env python3
"""
Defines function that performs the backward algorithm for a Hidden Markov Model
"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''
    Performs the backward algorithm for a hidden markov model
    '''
    try:
        T = Observation.shape[0]
        N, M = Emission.shape
        beta = np.zeros((N, T))
        beta[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for n in range(N):
                Transitions = Transition[n, :]
                Emissions = Emission[:, Observation[t + 1]]
                beta[n, t] = np.sum((Transitions * beta[:, t + 1]) * Emissions)

        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
        return P, beta
    except Exception:
        return None, None
