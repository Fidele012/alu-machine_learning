#!/usr/bin/env python3
"""
    Matrix multiplication
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Return the multiplication of two numpy matrices.
    """
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)
    return np.matmul(np_mat1, np_mat2)
