#!/usr/bin/env python3
"""Pooling"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function to perform pooling on images"""

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Gives dimensions of the output after pooling and considering stride
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, new_h, new_w, c))

    # Pooling operation based on mode
    for y in range(0, new_h * sh, sh):
        for x in range(0, new_w * sw, sw):
            if mode == 'max':
                output[:, y // sh, x // sw, :] = np.max(
                    images[:, y: y + kh, x: x + kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, y // sh, x // sw, :] = np.mean(
                    images[:, y: y + kh, x: x + kw, :], axis=(1, 2))
            else:
                raise ValueError("Mode should be either 'max' or 'avg'")

    return output
