"""
NEURAL NET UTILITIES

Stefan Wong 2017
"""

import numpy as np


def create_circle_data(N, D, K, theta=0.2):

    X = np.zeros((N*K, D))              # data
    y = np.zeros(N*K, dtype ='uint8')   # labels

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 2.0, N)
        t = np.linspace(0.0, 2 * np.pi, N)
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X, y

def create_spiral_data(N, D, K, theta=0.2):
    """
    INPUTS:
        N:
            Number of points per class
        D:
            Dimension of data
        K:
            Number of classes

    """

    X = np.zeros((N*K, D))          # Data matrix
    y = np.zeros(N*K, dtype='uint8')    # class labels

    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return X, y





