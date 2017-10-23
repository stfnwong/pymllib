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


def dropout_forward(X, dropout_param):
    """
    Compute the forward pass for (inverted) drop out
    """

    p = dropout_param['p']
    mode = dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*X.shape) < p) / p
        out = X * mask
    elif mode == 'test':
        mask = None
        out = X

    cache = (dropout_param, mask)
    out = out.astype(X.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Compute the backward pass for (inverted) dropout
    """

    dropout_param, mask = cache
    mode = dropout_param['mode']
    dx = None

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout

    return dx


def batchnorm_forward(X, gamma, beta, bn_param):
    """
    Forward pass for batch normalization
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = X.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=X.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=X.dtype))
    out = None
    cache = None

    if mode == 'train':
        # Training time forward pass

        # shape of mu is (D,)
        mu = 1 / float(N) * np.sum(X, axis=0)
        # shape of xmu is (N,D)
        xmu = (X - mu)**2
        # shape of var is (D,)
        var = 1 / float(N) * np.sum(xmu, axis=0)
        # shape sqrt is (D,)
        sqrtvar = np.sqrt(var + eps)
        invsqrt = 1. / sqrtvar
        xh = xmu * invsqrt
        out = gamma * xh + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var
        cache = (mu, (X - mu), xmu, var, sqrtvar, invsqrt,
                 (xmu * invsqrt), (gamma * xmu * invsqrt),
                 gamma, beta, X, bn_param)
    elif mode == 'test':
        # Test time forward pass
        mu = running_mean
        var = running_var
        xhat = (X - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Compute the backward pass for batch normalizationc
    """

    dx = None
    dgamma = None
    dbeta = None
    mu, xmu, square, var, sqrtvar, invvar, va2, va3, gamma, beta, X, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop
    # TODO ; Proper analysis of this...
    dva3 = dout
    dbeta = np.sum(dout, axis=0)
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar
    dsquare = 1 / float(N) * np.ones((square.shape)) * dvar
    dxmu += 2 * xmu * dsquare

    dx = dxmu
    dmu = -np.sum(dxmu, axis=0)
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta

# TODO : Alternative function?



# TODO ; Specialty layers

def affine_norm_relu_forward(X, v, b, gamma, beta, bn_param):
    """
    Performs an affine transform followed by a ReLU

    Inputs:
        - X:
            Input to the affine layer
        - w, b :
            Weights for the affine layer
        - gamma, beta :
            Weight for the batch norm regularization
        - bn_params :
            Parameters to the batchnorm layer
    """

def softmax_loss(X, y):
    """
    Compute loss and gradient for softmax classification
    """
    probs = np.exp(X - np.max(X, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = X.shape[0]


    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
