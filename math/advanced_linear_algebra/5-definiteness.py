#!/usr/bin/env python3
"""
Checks for the definiteness of a matrix and returns string
"""
import numpy as np


def definiteness(matrix):
    """
    returns a string about the definiteness
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        # If eigenvalue calculation fails, the matrix is not valid
        return None

    # Check for complex eigenvalues
    if np.iscomplexobj(eigenvalues):
        return None

    # Determine definiteness based on eigenvalues
    positive = np.all(eigenvalues > 0)
    negative = np.all(eigenvalues < 0)
    zero = np.any(np.isclose(eigenvalues, 0))

    if positive and not zero:
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif negative and not zero:
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
