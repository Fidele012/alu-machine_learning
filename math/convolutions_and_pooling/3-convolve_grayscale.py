#!/usr/bin/env python3
"""Strided Convolution"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Function to perform convolution with custom padding and stride"""

    if images.ndim == 4:
        m, h, w, _ = images.shape
        # Convert RGB to grayscale
        images = (0.2989 * images[:, :, :, 0] +
                  0.5870 * images[:, :, :, 1] +
                  0.1140 * images[:, :, :, 2])
    else:
        m, h, w = images.shape

    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Dimensions of the output after padding and considering stride
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    # Pad the grayscale images
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize the output array
    output = np.zeros((m, new_h, new_w))

    # Convolve the images with kernel
    for y in range(0, new_h * sh, sh):
        for x in range(0, new_w * sw, sw):
            output[:, y // sh, x // sw] = np.sum(
                padded[:, y: y + kh, x: x + kw] * kernel, axis=(1, 2))

    return output
