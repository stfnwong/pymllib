"""
ACTIVATIONS
Activation functions for neural networks and derivatives

Stefan Wong 2017
"""

import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_dz(z):
    d = np.zeros_like(z)
    d[z > 0] = 1

    return d

# Sigmoid non-linearity
def logistic(z):
    return 1./(1 + np.exp(-z))

def logistic_dz(z):
    p = logistic(z)
    return p * (1 - p)
