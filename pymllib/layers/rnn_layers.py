"""
RNN_LAYERS
Layer functions for recurrent neural networks
"""

import numpy as np


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Forward pass for a single timestep of a vanilla RNN that uses
    a tanh activation function. Input data has dimension D, hidden
    state has dimension H, each minibatch has size N.

    Inputs:
        - X      : Input data for this timestep. Shape (N, D)
        - prev_h : Hidden state from previous timestep. Shape (N, H)
        - Wx     : Weight matrix for input-to-hidden connections. Shape (D, H)
        - Wh     : Weight matrix for hidden-to-hidden connections. Shape (H, H)
        - b      : Biases. Shape (H,)

    Returns tuple of:
        - next_h : Next hidden state of shape (N, H)
        - cache  : Tuple of values needed for backward pass
    """

    forward = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    next_h = np.tanh(forward)
    cache = (x, Wx, prev_h, Wh, forward)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
        - dnext_h : Gradient of loss with respect to next
        hidden state.
        - cache : Tuple of (x, Ws, prev_h, Wh, forward)

    Returns a tuple of:
        - dx      : Gradients of input data. Shape (N, D)
        - dprev_h : Gradients of previous hidden state. Shape (N, H)
        - dWx     : Gradients of input-to-hidden weights. Shape (N, H)
        - dWh     : Gradients of hidden-to-hidden weights. Shape (H, H)
        - db      : Gradients of bias vector. Shape (H,)
    """

    x, Wx, prev_h, Wh, forward = cache
    # Backprop the forward pass
    dforward = (1 - np.tanh(forward)**2) * dnext_h

    dx = np.dot(dforward, Wx.T)
    dWx = np.dot(x.T, dforward)
    dprev_h = np.dot(dforward, Wh.T)
    dWh = np.dot(prev_h.T, dforward)
    db = np.sum(dforward, axis=0)

    return dx, dprev_h, dWx, dWh, db




def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function
    """

    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]

    return top / (1 + z)
