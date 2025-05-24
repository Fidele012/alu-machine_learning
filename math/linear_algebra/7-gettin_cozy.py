#!/usr/bin/env python3
"""
    A function that concatenates a 2d matix
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Return the concatenation of two 2D matrices.
    """
    if axis == 0:
        # Check if the number of columns match for vertical concatenation
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        # Check if the number of rows match for horizontal concatenation
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        # Invalid axis
        return None
