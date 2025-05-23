#!/usr/bin/env python3
"""
    A function that adds 2D matices
"""


def add_matrices2D(mat1, mat2):
    """
    Return the sum of two 2D matrices.
    """
    # check if the matrix is not of the same shape
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # loop through the matrix and add the elements
    new_matrix = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat1[0])):
            sum = mat1[i][j] + mat2[i][j]
            new_row.append(sum)
        new_matrix.append(new_row)
    return new_matrix
