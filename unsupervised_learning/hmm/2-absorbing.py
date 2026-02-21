#!/usr/bin/env python3
'''
    function def absorbing(P): that
    determines if a markov chain is absorbing
'''
import numpy as np


def absorbing(P):
    '''
        Determines if a markov chain is absorbing

        A Markov chain is absorbing if:
        1. It has at least one absorbing state (diagonal element = 1)
        2. From every state, it's possible to reach an absorbing state

        Args:
            P: square 2D numpy.ndarray of shape (n, n) representing
               the transition matrix

        Returns:
            True if the Markov chain is absorbing, False otherwise
            None if P is invalid
    '''
    # Validate input
    if not isinstance(P, np.ndarray):
        return None

    if len(P.shape) != 2:
        return None

    n1, n2 = P.shape
    if n1 != n2:
        return None

    n = n1

    # Get diagonal elements (absorbing states have P[i,i] = 1)
    D = np.diagonal(P)
    absorbing_states = (D == 1)

    # Check if there's at least one absorbing state
    if not absorbing_states.any():
        return False

    # If all states are absorbing, return True
    if absorbing_states.all():
        return True

    # Check reachability using BFS/DFS or matrix powers
    # For each non-absorbing state, check if it can reach an absorbing state

    # Method: Check if we can reach absorbing states from non-absorbing ones
    # Use repeated matrix multiplication (transitive closure)
    reachable = P > 0  # Boolean matrix of direct reachability

    # Compute transitive closure (paths of any length)

