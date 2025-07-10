#!/usr/bin/env python3
""" Performs convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function to convolve the image SAME"""
    if images.ndim == 4:
        m, h, w, _ = images.shape
        # Convert RGB to grayscale
        images = (0.2989 * images[:, :, :, 0] +
                  0.5870 * images[:, :, :, 1] +
                  0.1140 * images[:, :, :, 2])
    else:
        m, h, w = images.shape

    kh, kw = kernel.shape

    # Calculate the padding needed for 'SAME' convolution
    pad_h = (kh - 1) // 2 + (kh - 1) % 2
    pad_w = (kw - 1) // 2 + (kw - 1) % 2

    # Pad the grayscale images using zero padding
    padded = np.pad(images, ((0, 0), (pad_h, pad_h),
                             (pad_w, pad_w)), mode='constant')

    # Initialize an array to store the convolved images
    output = np.zeros((m, h, w))

    # Convolve each image with the kernel
    for y in range(h):
        for x in range(w):
            output[:, y, x] = np.sum(
                padded[:, y: y + kh, x: x + kw] * kernel, axis=(1, 2))

    return output
