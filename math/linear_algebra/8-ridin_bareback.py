#!/usr/bin/env python3
"""
    A function to multiyply two matrices
"""


def mat_mul(mat1, mat2):
    """
    Return the multiplication of two matrices.
    """
    # check if the matrix is not of the same shape
    if len(mat1[0]) != len(mat2):
        return None
    # loop through the matrix and add the elements
    new_matrix = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            mul = 0
            for k in range(len(mat2)):
                mul += mat1[i][k] * mat2[k][j]
            new_row.append(mul)
        new_matrix.append(new_row)
    return new_matrix
