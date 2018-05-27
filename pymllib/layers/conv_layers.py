"""
CONV_LAYERS
Convolutional Layers. Adapted (stolen) from CS231n
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pymllib.layers.im2col_cython import col2im_cython, im2col_cython
    from pymllib.layers.im2col_cython import col2im_6d_cython
except ImportError:
    print("Failed to import im2col_cython. Ensure that setup.py has")
    print("been run with build_ext --inplace.")
    print("eg: python3 setup.py build_ext --inplace")

from  pymllib.layers import im2col
import numpy as np
# Ordinary layers
from pymllib.layers import layers

def conv_forward_naive(X, w, b, conv_param):

    N, C, H, W = X.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    # Add padding to each image
    x_pad = np.pad(X, ((0,0), (0,0), (P,P), (P,P)), 'constant')
    # Size of the output
    Hh = int(1 + (H + 2 * P - HH) / S)
    Hw = int(1 + (W + 2 * P - WW) / S)
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
    F, _, HH, WW = w.shape
    _, _, Hh, Hw = dout.shape
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


# ======== FAST CONV LAYERS ======== #
def conv_forward_im2col(x, w, b, conv_param):
    """
    A fast implementation of the forward pass for a convolutional layer
    based on im2col and col2im
    """
    N, C, H, W = x.shape

    num_filters, _, filter_h, filter_w = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    # Dimension check
    assert (W + 2 * pad - filter_w) % stride == 0, 'Width does not align'
    assert (H + 2 * pad - filter_h) % stride == 0, 'Height does not align'
    # Create output
    out_height =int(1 + (H + 2 * pad - filter_h) / stride)
    out_width = int(1 + (W + 2 * pad - filter_w) / stride)
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)

    return out, cache

def conv_backward_im2col(dout, cache):
    """
    A fast implementation of the backward pass for a convolutional layer
    """

    x, w, b, conv_param, x_cols = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))
    dout_reshaped = dout.transpose(1, 0, 2, 3).rehsape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2img_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    assert (W + 2 * pad - WW) % stride == 0, 'Width does not align'
    assert (H + 2 * pad - HH) % stride == 0, 'Height does not align'

    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    #x_padded = x_padded.astype(np.int32)

    # Compute output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = int(1 + (H - HH) / stride)
    out_w = int(1 + (W - WW) / stride)
    # Perform im2col operation
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                               shape=shape,
                                               strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)
    # Now convolutions are just a large matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
    # reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)
    # Return a contiguous array.
    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)

    return out, cache


def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))
    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

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
    out_width =  int(1 + (W - pool_w) / stride)

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



# Spatial batchnorm

def spatial_batchnorm_forward(x, gamma, beta, bn_param):

    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        # Find average for each channel
        mu = (1.0 / (N * H * W) * np.sum(x, axis=(0, 2, 3))).reshape(1, C, 1, 1)
        var = (1.0 / (N * H * W) * np.sum((x - mu)**2, axis=(0, 2, 3))).reshape(1, C, 1, 1)

        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)

        running_mean = momentum * running_mean + (1.0 - momentum) * np.squeeze(mu)
        running_var = momentum * running_var + (1.0 - momentum) * np.squeeze(mu)

        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        cache = (mu, var, x, xhat, gamma, beta, bn_param)

    elif mode == 'test':
        mu = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)

        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode %s' % str(mode))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    mu, var, x, xhat, gamma, beta, bn_param = cache
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)

    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)
    # backprop
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * xhat, axis=(0, 2, 3))

    Nt = N * H * W
    dx = (1.0 / Nt) * gamma * (var + eps)**(-1.0 / 2.0) * (Nt * dout - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1) - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1))

    return dx, dgamma, dbeta

# Combinational convolution functions
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a relu, and a pool.

    TODO : docstring
    """

    # TODO : implement Cython versions
    #a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    a, conv_cache = conv_forward_strides(x, w, b, conv_param)
    s, relu_cache = layers.relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)

    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv_relu_pool layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = layers.relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_strides(da, conv_cache)
    #dx, dw, db = conv_backward_naive(da, conv_cache)

    return dx, dw, db


def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):

    conv, conv_cache = conv_forward_strides(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    relu, relu_cache = layers.relu_forward(norm)
    out, pool_cache  = max_pool_forward_fast(relu, pool_param)

    cache = (conv_cache, norm_cache, relu_cache, pool_cache)

    return out, cache


def conv_norm_relu_pool_backward(dout, cache):

    conv_cache, norm_cache, relu_cache, pool_cache = cache

    dpool = max_pool_backward_fast(dout, pool_cache)
    drelu = layers.relu_backward(dpool, relu_cache)
    dnorm,dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_strides(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta
