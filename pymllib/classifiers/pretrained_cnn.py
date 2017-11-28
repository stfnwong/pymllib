"""
PRETRAINED CNN
From CS231n
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import h5py
# Layers
from pymllib.layers import layers


class PretrainedCNN(object):
    def __init__(self, **kwargs):
        """
        This is the pre-trained CNN from CS231n
        """
        self.dtype = kwargs.pop('dtype', np.float32)
        self.verbose = kwargs.pop('verbose', False)
        self.num_classes = kwargs.pop('num_classes', 100)
        self.input_size = kwargs.pop('input_size', 64)
        h5file = kwargs.pop('h5file', None)

        # Set up convolution parameters
        self.conv_params = []
        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})

        self.filter_sizes = [5, 3, 3, 3, 3, 3, 3, 3, 3]
        self.num_filters = [64, 64, 128, 128, 256, 256, 512, 52, 1024]
        hidden_dim = 512

        self.bn_params = []

        cur_size = input_size
        prev_dim = 3
        self.params = {}
        # Add conv layers
        for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
            fan_in = f * f * prev_dim
            self.params['W%d' % (i+1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
            self.params['b%d' % (i+1)] = np.zeros(next_dim)
            self.params['gamma%d' % (i+1)] = np.ones(next_dim)
            self.params['beta%d' % (i+1)] = np.zeros(next_dim)
            self.bn_params.append({'mode': 'train'})
            prev_dim = next_dim
            if self.conv_params[i]['stride'] == 2:
                cur_size /= 2

        # Add fully connected layers
        fan_in = cur_size * cur_size * self.num_filters[-1]
        self.params['W%d' % (i+2)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
        self.params['b%d' % (i+2)] = np.zeros(hidden_dim)
        self.params['gamma%d' % (i+2)] = np.ones(hidden_dim)
        self.params['beta%d' % (i+2)] = np.zeros(hidden_dim)
        self.bn_params.append({'mode': 'train'})
        self.params['W%d' % (i+3)] = np.sqrt(2.0 / fan_in) * np.random.randn(hidden_dim, num_classes)
        self.params['b%d' % (i+3)] = np.zeros(num_classes)

        # Cast to correct type
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        if h5file is not None:
            self.load_weights(h5_file)

    def load_weights(self, h5_file):
        """
        Load pre-trained weights from an HDF5 file.
        """

        # Before loading weights, make a dummy forward pass
        # to initialize the running averages in the parameters
        x = np.random.randn(1, 3, self.input_size, self.input_size)
        y = np.random.randint(self.num_classes, size=1)
        loss, grads = self.loss(x, y)

        with h5py.File(h5_file, 'r') as f:
            for k, v in f.items():
                v = np.asarray(v)
                if k in self.params:
                    if self.verbose:
                        print('%s : %s | param %s : %s' % k, v.shape, k, self.params[k].shape)
                    if v.shape == self.params[k].shape:
                        self.params[k] = v.copy()
                    elif v.T.shape == self.params[k].shape:
                        self.params[k] = v.T.copy()
                    else:
                        raise ValueError("Shapes for %s do not match" % k)

                if k.startswith('running_mean'):
                    i = int(k[12:]) - 1
                    if self.bn_params[i]['running_mean'].shape != v.shape:
                        raise ValueError("bn_param[%d][running_mean].shape does not match %s shape" % (i, v))
                    self.bn_params[i]['running_mean'] = v.copy()
                    if self.verbose:
                        print("%s : %s" % (k, v.shape))

                if k.startswith('running_var'):
                    i = int(k[11:]) - 1
                    if self.bn_params[i]['running_var'].shape != v.shape:
                        raise ValueError("bn_param[%d][running_var].shape does not match %s shape" % (i, v))
                    self.bn_params[i]['running_var'] = v.copy()
                    if self.verbose:
                        print("%s : %s" % (k, v.shape))

            # Cast datatypes
            for k, v in self.params.items():
                self.params[k] = v.astype(self.dtype)

    def forward(self, X, start=None, end=None, mode='test'):
        """
        TODO : complete docstring
        """
        X = X.astype(self.dtype)
        if start is None:
            start = 0
        if end is None:
            end = len(self.conv_params) + 1

        layer_caches = []
        prev_a = X
        for i in range(start, end + 1):
            i1 = i + 1
            w = self.params['W%d' % i1]
            b = self.params['b%d' % i1]
            gamma = self.params['gamma%d' % i1]
            beta = self.params['beta%d' % i1]
            conv_param = self.conv_params[i]
            bn_param = self.bn_params[i]
            bn_param['mode'] = mode
            if 0 <= i < len(self.conv_params):
                # Conv layer
                next_a, cache = layers.conv_bn_relu_forward(prev_a, w, b,
                                                            gamma, beta,
                                                            conv_param, bn_param)
            elif i == len(self.conv_params):
                # Affine layer
                next_a, cache = layers.affine_norm_relu_forward(prev_a, w, b, gamma, beta, bn_param)
            elif i == len(self.conv_params) + 1:
                # Score layer
                next_a, cache = layers.affine_forward(prev_a, w, b)
            else:
                raise ValueError('Invalid layer index %d' % i)

            layer_caches.append(cache)  # TODO : re-implement as dict
            prev_a = next_a

        out = prev_a
        cache = (start, end, layer_caches)

        return out, cache

    def backward(self, dout, cache):
        """
        TODO : docstring
        """

        start, end, layer_caches = cache
        dnext_a = dout
        grads = {}

        for i in reversed(range(start, end + 1)):
            i1 = i + 1
            if i1 == len(self.conv_params) + 1:
                # Last affine layer
                dprev_a, dw, db = layers.affine_backward(dnext_a, layer_caches.pop())
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
            elif i == len(self.conv_params):
                # Affine hidden layer
                dprev_a, dw, db, dgamma, dbeta = layers.affine_norm_relu_backward(
                    dnext_a, layer_caches.pop())
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
                grads['gamma%d' % i1] = dgamma
                grads['beta%d' % i1] = dbeta
            elif 0 <= i < len(self.conv_params):
                dprev_a, dw, db, dgamma, dbeta = layers.conv_bn_relu_backward(
                    dnext_a, layer_caches.pop())
                grads['W%d' % i1] = dw
                grads['b%d' % i1] = db
                grads['gamma%d' % i1] = dgamma
                grads['beta%d' % i1] = dbeta
            else:
                raise ValueError('Invalid layer index %d' % i)
            dnext_a = dprev_a

        dX = dnext_a

        return dX, grads



    def loss(X, y=None):
        """
        Classification loss used to train the network.

        Inputs:
            - X : Array of data of shape (N, 3, 64, 64)
            - y : Array of labels of shape (N,)

        """

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        scores, cache = self.forward(X, mode=mode)
        if mode == 'test':
            return scores

        loss, dscores = layers.softmax_loss(scores, y)
        dx, grads = self.backward(dscores, cache)

        return loss, grads
