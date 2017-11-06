"""
LAYERS
Functional implementation of layers in neural network. These are based on the
layers in Caffe.
"""

import numpy as np
import im2col
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

# TODO : Other loss functions, eg, for autoencoder...

# ==== CONVOLUTION LAYERS ==== #
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
        print("(Conv backward naive) : Computing image %d" % nprime)
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(w[f, :, :, :])
                            mask2 = np.zeros_like(w[f, :, :, :])
                            # NOTE TO SELF
                            # The mask here is supposed to represent the fact
                            # that all the components where nprime != n and
                            # mprime != m become zero in the partial derivative
                            if (i + P - k * S) < HH and (i + P - k * S) >= 0:
                                mask1[:, i + P - k * S, :] = 1.0
                            if (j + P - l * S) < WW and (j + P- l * S) >= 0:
                                mask2[:, :, j + P - l * S] = 1.0
                            w_masked = np.sum(w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
                            dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked

    return dx, dw, db


# Util layer forward passes
def max_pool_forward_reshape(x, pool_param):
    """
    Fast implementation of max pooling that uses some clever reshaping
    """

    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    assert pool_w == pool_h == stride, 'Invalid pool param'
    assert H % pool_h == 0
    assert W % pool_w == 0

    x_reshaped = x.reshape(N, C, int(H / pool_h), pool_h, int(W / pool_w), pool_w)
    out = x_reshaped.max(axis=3).max(axis=4)
    cache = (x, x_reshaped, out)

    print("(max_pool_forward_reshape), out.shape")
    print(out.shape)

    return out, cache


def max_pool_forward_im2col(x, pool_param):
    """
    An implementation of the forward pass for maxpooling based
    on im2col. Not much faster than the naive version

    """
    N, C, W, H = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_h) % stride == 0, "Invalid height"
    assert (W - pool_w) % stride == 0, "Invalid width"

    out_height = int(1 + (H - pool_h) / stride)
    out_width = int(1 + (W - pool_w) / stride)

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_h, pool_w, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)

    return out, cache


def max_pool_forward_fast(x, pool_param):
    """
    Fast implementation of the max pool forward pass

    """
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    same_size = (pool_w == pool_h == stride)
    tiles = (H % pool_h == 0) and (W % pool_w == 0)

    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im2col_cache)

    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    A fast implementation of the backward pass for a max pool layer


    """
    method, real_cache = cache
    if method == 'reshape':
        return max_pool_backward_reshape(dout, real_cache)
    elif method == 'im2col':
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unknown method %s' % method)


# Util layer backward passes
def max_pool_backward_reshape(dout, cache):
    """
    Fast implementation of the max_pool_reshape backward pass. This
    function can only be used if the forward pass was computed using
    max_pool_forward_reshape
    """

    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)

    # TODO : for some reason dout is smaller than I expected....
    print(dout.shape)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3,5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


def max_pool_backward_im2col(dout, cache):
    """
    An implementation of the backward pass for max pooling based on
    im2col.

    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
    dx = im2col.col2im_indicies(dx_cols, (N * C, 1, H, W), pool_h, pool_w, padding=0, stride=stride)
    dx = dx.reshape(x.shape)

    return dx


# Combinational convolution functions
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a relu, and a pool.

    TODO : docstring
    """

    # TODO : implement Cython versions
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)

    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv_relu_pool layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)

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



# ===== LAYER OBJECT ===== #

class Layer(object):
    def __init__(self, input_dim, hidden_dim, weight_scale=1e-2):
        self.W = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.b = np.zeros(hidden_dim)
        self.input_cache = None


class AffineLayer(Layer):
    def __str__(self):
        s = []
        s.append('Affine Layer:\n\t (%d x %d)\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        self.input_cache = X
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        x2 = np.reshape(X, (N,D))
        print('affine x2 shape (%d, %d)' % (x2.shape[0], x2.shape[1]))
        out = np.dot(x2, self.W) + self.b

        return out

    def backward(self, dout):
        """
        Compute the backward pass for an affine layer
        """
        X = self.input_cache
        dx = np.dot(dout, self.W.T).reshape(X.shape)
        dw = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
        db = np.sum(dout, axis=0)

        return dx, dw, db


class ReLULayer(Layer):
    def __str__(self):
        s = []
        s.append('ReLU Layer:\n\t (%d x %d)\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        """
        Computes the forward pass for a layer of rectified linear units
        """
        self.input_cache = X
        out = np.maximum(0, X)

        return out

    def backward(self, dout):
        """
        Compute the backward pass for a layer of rectified linear units
        """
        X = self.input_cache
        dx = np.array(dout, copy=True)
        dx[X <= 0] = 0

        return dx


class SoftmaxLayer(Layer):
    def __str__(self):
        s = []
        s.append('Softmax Layer:\n\t (%d x %d)\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        pass
