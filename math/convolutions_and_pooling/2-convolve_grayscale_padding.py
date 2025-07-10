#!/usr/bin/env python3
"""Convolution with Padding"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Function to perform convolution with custom padding """

    if images.ndim == 4:
        m, h, w, _ = images.shape
        # Convert RGB to grayscale
        images = (0.2989 * images[:, :, :, 0] +
                  0.5870 * images[:, :, :, 1] +
                  0.1140 * images[:, :, :, 2])
    else:
        m, h, w = images.shape

    kh, kw = kernel.shape
    ph, pw = padding

    # zero padding with custom padding values
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calculate the dimensions of the output
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    # Initialize the array for convolved images
    output = np.zeros((m, new_h, new_w))

    # Perform the convolution
    for y in range(new_h):
        for x in range(new_w):
            output[:, y, x] = np.sum(
                padded[:, y: y + kh, x: x + kw] * kernel, axis=(1, 2))

    return output
