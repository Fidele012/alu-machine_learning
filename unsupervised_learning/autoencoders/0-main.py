#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

autoencoder = __import__('0-vanilla').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# Test if encoder, decoder, and auto properties exist and are not None
np.random.seed(0)
encoder, decoder, auto = autoencoder(784, [128, 64], 32)
print(encoder is not None)  # Should print True
print(decoder is not None)  # Should print True
print(auto is not None)     # Should print True
