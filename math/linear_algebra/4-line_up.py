#!/usr/bin/env python3
"""
A function that adds two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Return the sum of two arrays.
    """
    # check if the array is not of the same shape
    if len(arr1) != len(arr2):
        return None
    # loop through the array and add the elements
    new_array = []
    for i in range(len(arr1)):
        sum = arr1[i] + arr2[i]
        new_array.append(sum)
    return new_array
