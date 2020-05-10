"""
LAYERS
Functional implementation of layers in neural network. These are based on the
layers in Caffe.
"""

import numpy as np
from typing import Any
from typing import Dict
from typing import Tuple


try:
    from pymllib.layers.im2col_cython import col2im_cython, im2col_cython
    from pymllib.layers.im2col_cython import col2im_6d_cython
except ImportError:
    print("Failed to import im2col_cython. Ensure that setup.py has")
    print("been run with build_ext --inplace.")
    print("eg: python3 setup.py build_ext --inplace")

from  pymllib.layers import im2col
import numpy as np

# Debug
#from pudb import set_trace; set_trace()

def affine_forward(X:np.ndarray, w:np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward pass for an affine layer
    """
    N = X.shape[0]
    D = np.prod(X.shape[1:])
    x2 = np.reshape(X, (N,D))
    out = np.dot(x2, w) + b
    cache = (X, w, b)

    return (out, cache)


def affine_backward(dout:np.ndarray, cache:tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the backward pass for an affine layer
    """
    X, w, b = cache
    dx = np.dot(dout, w.T).reshape(X.shape)
    dw = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)

    return (dx, dw, db)


def relu_forward(X:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward pass for a layer of rectified linear units
    """
    out = np.maximum(0, X)
    cache = X

    return (out, cache)


def relu_backward(dout:np.ndarray, cache:np.ndarray) -> np.ndarray:
    """
    Compute the backward pass for a layer of rectified linear units
    """
    X = cache
    dx = np.array(dout, copy=True)
    dx[X <= 0] = 0

    return dx


def dropout_forward(X:np.ndarray, dropout_param:dict) -> Tuple[Any, Tuple[Any, Any]]:
    """
    Compute the forward pass for (inverted) drop out
    """

    p    = dropout_param['p']
    mode = dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        mask = (np.random.rand(*X.shape) < p) / p
        out = X * mask
    elif mode == 'test':
        mask = None
        out = X
    else:
        raise ValueError('Invalid mode [%s]' % str(mode))

    cache = (dropout_param, mask)
    out = out.astype(X.dtype, copy=False)

    return (out, cache)


def dropout_backward(dout:np.ndarray, cache:tuple) -> np.ndarray:
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


def batchnorm_forward(X:np.ndarray,
                      gamma:float,
                      beta:float,
                      bn_param:dict) -> Tuple[np.ndarray, Any]:
#-> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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

    # NOTE: the caching here is a nightmare for type hinting...
    if mode == 'train':
        # Training time forward pass
        # Step 1
        mu = 1 / float(N) * np.sum(X, axis=0)
        # Step 2
        xmu = (X - mu)
        # Step 3
        sq = xmu**2
        # Step 4
        var = 1 / float(N) * np.sum(sq, axis=0)
        # Step 5
        sqvar = np.sqrt(var + eps)
        # Step 6
        invvar = 1.0 / sqvar
        # Step7
        va2 = xmu * invvar
        # Step 8
        va3 = gamma * va2
        # Step 9
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, sq, var, sqvar, invvar, va2, va3, gamma, beta, X, bn_param)
    elif mode == 'test':
        # Test time forward pass
        mu = running_mean
        var = running_var
        xhat = (X - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)
    else:
        raise ValueError("Invalid forward batchnorm mode [%s]" % mode)

    return (out, cache)


def batchnorm_backward(dout:np.ndarray, cache:tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    dva3     = dout
    dbeta    = np.sum(dout, axis=0)
    dva2     = gamma * dva3
    dgamma   = np.sum(va2 * dva3, axis=0)
    dxmu     = invvar * dva2
    dinvvar  = np.sum(xmu * dva2, axis=0)
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar
    dvar     = 0.5 * (var + eps)**(-0.5) * dsqrtvar
    dsquare  = 1 / float(N) * np.ones((square.shape)) * dvar
    dxmu += 2 * xmu * dsquare

    dx  = dxmu
    dmu = -np.sum(dxmu, axis=0)
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return (dx, dgamma, dbeta)


# ==== Convenience layers
# NOTE: output type hint should be Tuple[np.ndarray, Tuple[Tuple, Tuple]] or
# something....
def affine_relu_forward(X:np.ndarray, w:np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    Affine transform followed by ReLU, forward pass
    """
    a, fc_cache = affine_forward(X, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)

    return (out, cache)


def affine_relu_backward(dout:np.ndarray, cache:tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Affine transform followed by ReLU, backward pass
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return (dx, dw, db)


def affine_norm_relu_forward(X:np.ndarray,
                             w:np.ndarray,
                             b:np.ndarray,
                             gamma:float,
                             beta:float,
                             bn_param:dict) -> Tuple[np.ndarray, np.ndarray]:
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
    h, h_cache = affine_forward(X, w, b)
    hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    hnormrelu, relu_cache = relu_forward(hnorm)
    cache = (h_cache, hnorm_cache, relu_cache)

    return (hnormrelu, cache)


def affine_norm_relu_backward(dout:np.ndarray, cache:tuple) -> Tuple[np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray]:

    h_cache, hnorm_cache, relu_cache = cache
    dnormrelu = relu_backward(dout, relu_cache)
    dnorm, dgamma, dbeta = batchnorm_backward(dnormrelu, hnorm_cache)
    dx, dw, db, = affine_backward(dnorm, h_cache)

    return (dx, dw, db, dgamma, dbeta)


def softmax_loss(X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute loss and gradient for softmax classification
    """
    probs = np.exp(X - np.max(X, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = X.shape[0]

    l1 = np.log(probs[np.arange(N), y])
    loss = -np.sum(l1) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return (loss, dx)


# Sigmoid functions
def sigmoid_forward(X:np.ndarray, w:np.ndarray, b:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward pass of sigmoid function
    """

    N = X.shape[0]
    D = np.prod(X.shape[1:])
    x2 = np.reshape(X, (N,D))
    z = np.dot(x2, w) + b
    out = 1 / (1 + np.exp(-z))
    cache = (X, w, b)

    return (out, cache)


def sigmoid_backward(dout:np.ndarray, cache:Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute backward pass of sigmoid function
    """

    X, w, b = cache
    sf, _ = sigmoid_forward(X, w, b)
    #ddot = (1 - sf) * sf
    dx = np.dot(sf * (1-sf), w.T)
    dw = np.dot(X.T, sf * (1-sf))
    db = 1.0
    #dx = np.dot(ddot, w.T).reshape(X.shape)
    #dw = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, ddot)

    return (dx, dw, db)
