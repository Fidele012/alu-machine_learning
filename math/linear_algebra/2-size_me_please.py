#!/usr/bin/env python3
"""
A function that returns the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Returns:
    list: A list representing the shape of the matrix, where each element
          corresponds to the size of each dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
