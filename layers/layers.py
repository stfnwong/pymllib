"""
LAYERS
Functional implementation of layers in neural network. These are based on the
layers in Caffe.
"""

import numpy as np
# Debug
#from pudb import set_trace; set_trace()

def affine_forward(X, w, b):
    """
    Compute forward pass for an affine layer
    """
    N = X.shape[0]
    D = np.prod(X.shape[1:])
    x2 = np.reshape(X, (N,D))
    out = np.dot(x2, w) + b
    cache = (X, w, b)

    return out, cache

def affine_backward(dout, cache):
    """
    Compute the backward pass for an affine layer
    """
    X, w, b = cache
    dx = np.dot(dout, w.T).reshape(X.shape)
    dw = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def relu_forward(X):
    """
    Computes the forward pass for a layer of rectified linear units
    """
    out = np.maximum(0, X)
    cache = X

    return out, cache

def relu_backward(dout, cache):
    """
    Compute the backward pass for a layer of rectified linear units
    """
    X = cache
    dx = np.array(dout, copy=True)
    dx[X <= 0] = 0

    return dx


def softmax_loss(X, y):
    """
    Compute loss and gradient for softmax classification
    """
    probs = np.exp(X - np.max(X, axis=1, keepdims=True)
