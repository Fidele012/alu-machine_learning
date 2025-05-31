#!/usr/bin/env python3
"""
Module for calculating minor matrices
"""


def get_matrix_minor(matrix, i, j):
    """Returns the minor matrix after removing row i and column j"""
    return [[matrix[row][col] for col in range(len(matrix[row])) if col != j]
            for row in range(len(matrix)) if row != i]


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        det += ((-1) ** j) * matrix[0][j] * determinant(get_matrix_minor
                                                        (matrix, 0, j))
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Parameters:
        matrix: List of lists representing a square matrix

    Returns:
        Minor matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix or any row is empty
    if not matrix or not matrix[0] or not all(row for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle 1x1 matrix case
    if len(matrix) == 1:
        return [[1]]

    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Get minor matrix for position (i,j)
            minor_ij = get_matrix_minor(matrix, i, j)
            # Calculate determinant of minor matrix
            minor_row.append(determinant(minor_ij))
        minor_matrix.append(minor_row)

    # Verify output is not empty
    if not minor_matrix or not minor_matrix[0]:
        raise ValueError("matrix must be a non-empty square matrix")

    return minor_matrix
