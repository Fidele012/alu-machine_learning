#!/usr/bin/env python3
""" Performs convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function to convolve the image"""
    if images.ndim == 4:
        m, h, w, _ = images.shape
        # Convert RGB to grayscale
        images = (0.2989 * images[:, :, :, 0] +
                  0.5870 * images[:, :, :, 1] +
                  0.1140 * images[:, :, :, 2])
    else:
        m, h, w = images.shape

    kh, kw = kernel.shape

    # Calculate new dimensions after VALID convolution
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize the array for convolved images
    output = np.zeros((m, new_h, new_w))

    # Perform the convolution
    for i in range(new_h):
        for j in range(new_w):
            sub_matrix = images[:, i: i + kh, j: j + kw] * kernel
            output[:, i, j] = np.sum(sub_matrix, axis=(1, 2))
    return output
