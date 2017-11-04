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

    TODO: This does a reshape for minibatches, include in
    docstring
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
        raise ValueError("Invalid forward batchnorm mode %s" % mode)

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


# ==== Convenience layers
def affine_relu_forward(X, w, b):
    """
    Affine transform followed by ReLU, forward pass
    """
    a, fc_cache = affine_forward(X, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)

    return out, cache

def affine_relu_backward(dout, cache):
    """
    Affine transform followed by ReLU, backward pass
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db

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

#    print("probs.shape; (%d, %d)" % (probs.shape[0], probs.shape[1]))
#    if np.min(probs) < 0.0:
#        print('min of probs is %f' % np.min(probs))
#
    #l1 = np.log(np.max(probs[np.arange(N), y], 1e-15))
    l1 = np.log(probs[np.arange(N), y])
    loss = -np.sum(l1) / N
    #loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx

# ==== CONVOLUTION FUNCTIONS ==== #
def conv_forward_naive(X, w, b, conv_param):

    N, C, H, W = X.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    # Add padding to each image
    x_pad = np.pad(X, ((0,0), (0,0), (P,P), (P,P)), 'constant')
    # Size of the output
    Hh = 1 + (H + 2 * P - HH) / S
    Hw = 1 + (W + 2 * P - WW) / S
    Hh = int(Hh)
    Hw = int(Hw)
    out = np.zeros((N, F, Hh, Hw))

    for n in range(N):      # Iterate over images
        for f in range(F):  # Iterate over kernels
            for k in range(Hh):
                for l in range(Hw):
                    pad = x_pad[n, :, k * S:k * S + HH, l * S:l * S + WW]
                    out[n, f, k, l] = np.sum(pad * w[f, :, : , :]) + b[f]
                    #out[n, f, k, l] = np.sum(pad * w[f, :]) + b[f]
    cache = (X, w, b, conv_param)

    return out, cache

def conv_backward_naive(dout, cache):
    """
    Naive implementation of backward pass for convolutional layer
    """

    dx = None
    dw = None
    db = None

    # TODO : A better name for the weight param?
    X, w, b, conv_param = cache
    P = conv_param['pad']
    x_pad = np.pad(X, ((0,0), (0,0), (P,P), (P,P)), 'constant')

    N, C, H, W = X.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Hw = dout.shape
    S = conv_param['stride']

    # Weights
    dw = np.zeros((F, C, HH, WW))
    for fprime in range(F):
        for cprime in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_xpad = x_pad[:, cprime, i:i + Hh * S:S, j:j + Hw * S:S]
                    dw[fprime, cprime, i, j] = np.sum(dout[:, fprime, :, :] * sub_xpad)
    # Biases
    db = np.zeros((F))
    for fprime in range(F):
        db[fprime] = np.sum(dout[:, fprime, :, :])

    # "Downstream" (data) gradients
    dx = np.zeros((N, C, H, W))
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            if (i + P - k *S) < HH and (i + P - k * S) >= 0:
                                mask1[:, i + P - k * S, :] = 1.0
                            if (j + P - l * S) < WW and (j + P- l * S) >= 0:
                                mask2[:, :, j + P - l * S] = 1.0
                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

    return dx, dw, db

# Sigmoid functions
def sigmoid_forward(X, w, b):
    """
    Compute forward pass of sigmoid function
    """

    N = X.shape[0]
    D = np.prod(X.shape[1:])
    x2 = np.reshape(X, (N,D))
    z = np.dot(x2, w) + b
    out = 1 / (1 + np.exp(-z))
    cache = (X, w, b)

    return out, cache

def sigmoid_backward(dout, cache):
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

    return dx, dw, db






