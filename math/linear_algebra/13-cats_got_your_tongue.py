#!/usr/bin/env python3
"""
    concatenates a n array
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Return the concatenation of two numpy matrices.
    """
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)
    return np.concatenate((np_mat1, np_mat2), axis=axis)
