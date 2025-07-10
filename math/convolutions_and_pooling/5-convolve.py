#!/usr/bin/env python3
"""Multiple Kernel Convolution"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function to perform convolution with multiple kernels"""

    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Check if channels in images match channels in kernel
    if c != kc:
        raise ValueError(
            "Number of channels in the image and kernel should be the same")

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Gives dimensions of the output after padding and considering stride
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1

    # Pad the images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Initialize the output array
    output = np.zeros((m, new_h, new_w, nc))

    # Convolve the images with multiple kernels
    for n in range(nc):
        for y in range(0, new_h * sh, sh):
            for x in range(0, new_w * sw, sw):
                output[:, y // sh, x // sw, n] = np.sum(
                    padded[:, y: y + kh, x: x + kw, :] * kernels[:, :, :, n],
                    axis=(1, 2, 3))

    return output
