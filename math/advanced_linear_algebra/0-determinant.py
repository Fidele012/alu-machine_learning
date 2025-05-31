#!/usr/bin/env python3
"""
Module for calculating matrix determinants
"""


def get_minor(matrix, i, j):
    """Returns the minor matrix after removing row i and column j"""
    return [[matrix[row][col] for col in range(len(matrix[row])) if col != j]
            for row in range(len(matrix)) if row != i]


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Parameters:
        matrix: List of lists representing a square matrix

    Returns:
        Determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle empty matrix case [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Check if matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # 1x1 matrix case
    if len(matrix) == 1:
        return matrix[0][0]

    # 2x2 matrix case
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # For larger matrices, use Laplace expansion along first row
    det = 0
    for j in range(len(matrix)):
        det += ((-1) ** j) * matrix[0][j] * determinant(get_minor
                                                        (matrix, 0, j))

    return det
