"""
AUTOENCODER
Attempt to use an fcnet as an autoencoder

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pymllib.layers.layers as layers

# Debug
from pudb import set_trace; set_trace()

def sparse_autoencoder_loss(X, y, rho, beta=1.0, sparsity=0.05):

    hs_loss, hs_dx = layers.half_square_loss(X, y)
    # Apply sparsity penalty
    s1 = sparsity
    s2 = (1.0 - sparsity)
    p_loss = np.sum(s1 * np.log(s1 / rho) + s2 * np.log(s2 / (1 - rho)), axis=0)
    #p_loss = np.sum(s1 * np.log(s1 / rho) + s2 * np.log(s2 / (1 - rho)), axis=0, keepdims=True)
    # final loss
    loss = hs_loss + (beta * p_loss)

    return loss, hs_dx

def stable_softmax(X):
    exps = np.exp(X) - np.max(X)
    loss = exps/ np.sum(exps)

    return loss

def auto_affine_backward(dout, cache, rho, beta=1.0, s=0.05):
    """
    Compute the backward pass for an affine layer
    """
    X, w, b = cache
    sparse_dout = beta * (-(s / rho) + ((1 - s) / (1 - rho))) * dout
    dx = np.dot(sparse_dout, w.T).reshape(X.shape)
    dw = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


# TODO : need to check the layer shapes here c
class Autoencoder(object):
    def __init__(self, hidden_dims, input_dim, **kwargs):
                 #dropout=0, use_batchnorm=False, reg=0.0,
                 #weight_scale=1e-2, dtype=np.float32, seed=None,
                 #verbose=False):
        """
        AUTOENCODER
        An implementation of an autoencoder using the same architecture
        as other models in this library
        """
        if type(hidden_dims) is not list:
            raise ValueError("hidden_dims must be a list")

        self.verbose       = kwargs.pop('verbose', False)
        self.use_batchnorm = kwargs.pop('use_batchnorm', False)
        self.dropout       = kwargs.pop('dropout', 0)
        self.weight_scale  = kwargs.pop('weight_scale', 1e-2)
        self.weight_init   = kwargs.pop('weight_init', 'gauss_sqrt')
        self.reg           = kwargs.pop('reg', 0.0)
        self.dtype         = kwargs.pop('dtype', np.float32)
        self.seed          = kwargs.pop('seed', None)
        self.num_layers    = 1 + len(hidden_dims)       # The FCNet formuation is 1+len(hidden_dims)
        self.params        = {}
        self.use_dropout   = self.dropout > 0

        # Init the params of the network into the dictionary self.params
        dims = [input_dim] + hidden_dims + [input_dim]
        print('DEBUG: len(dims) = %d' % len(dims))

        # Init weights
        #Ws = {}
        #bs = {}
        if self.verbose:
            print('Dim pairs:')
        # generate layers
        for i in range(len(dims)-1):
            W = self._weight_init(dims[i], dims[i+1])
            b = np.zeros(dims[i+1])

            self.params.update({'W' + str(i+1): W,
                                'b' + str(i+1): b})

        #for i in range(len(dims)):
        #    # Debug : remove
        #    #if self.verbose:
        #    #    print('W%d : (%d %d)' % (i+1, dims[i], dims[i+1]))

        #    print(i)
        #    if i == (len(dims)-1):
        #        Ws['W' + str(i+1)] = self._weight_init(dims[i], 1)
        #        bs['b' + str(i+1)] = np.zeros(1)
        #    else:
        #        Ws['W' + str(i+1)] = self._weight_init(dims[i], dims[i+1])
        #        bs['b' + str(i+1)] = np.zeros(dims[i+1])

        #self.params.update(bs)
        #self.params.update(Ws)

        # Cast params to correct data type
        if self.verbose:
            print("Casting parameters to type %s" % str(self.dtype))
        for k,v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def __str__(self):
        s = []
        for l in range(self.num_layers):
            wl = self.params['W' + str(l+1)]
            bl = self.params['b' + str(l+1)]
            s.append('Layer %d\n\t W%d: (%d, %d),\t b%d: (%d)\n' % (l+1, l+1, wl.shape[0], wl.shape[1], l+1, bl.shape[0]))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def _weight_init(self, N, D, fsize=None):
        """
        WEIGHT_INIT
        Set up the weights for a given layer.
        """
        if self.weight_init == 'gauss':
            if fsize is None:
                W = self.weight_scale * np.random.randn(N, D)
            else:
                W = self.weight_scale * np.random.randn(N, D, fsize, fsize)
        elif self.weight_init == 'gauss_sqrt':
            if fsize is None:
                W = self.weight_scale * np.random.randn(N, D) * (1 / np.sqrt(2.0 / (N+D)))
            else:
                W = self.weight_scale * np.random.randn(N, D, fsize, fsize) * (1 / np.sqrt(2.0 / (N+D)))
        elif self.weight_init == 'xavier':
            w_lim = 2 / np.sqrt(N + D)
            if fsize is None:
                wsize = (N, D)
            else:
                wsize = (N, D, fsize, fsize)
            W = np.random.uniform(low=-w_lim, high=w_lim, size=wsize)
        else:
            raise ValueError('Invalid weight init method %s' % self.weight_init)

        return W

    # TODO : This is for debugging only, remove
    def print_weight_sizes(self):
        s = []
        for k, v in sorted(self.params.items()):
            if k[:1] == 'W':
                s.append("%s : %s\n" % (k, v.shape))

        return ''.join(s)

    def loss(self, X, y=None):
        """
        LOSS
        Compute forward and backward pass and produce loss and gradients
        """
        X = X.astype(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        y = np.reshape(y, (y.shape[0], np.prod(y.shape[1:]))).T
        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

        # Forward pass with activations
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            # TODO : Note that in general these will need to be smaller
            # than either the input or the output
            h, cache_h = layers.affine_forward(h, w, b)
            hidden['h' + str(idx)] = h
            hidden['cache_h' + str(idx)] = cache_h

        scores = hidden['h' + str(self.num_layers)]
        print('scores.shape : %s' % str(scores.shape))
        print('y.shape      : %s' % (str(y.shape)))
        # TODO: debug, remove
        for k in range(self.num_layers):
            print('%s : %s' % ('h' + str(k), str(hidden['h' + str(k)].shape)))

        if mode == 'test':
            return scores

        loss = 0.0
        grads = {}
        data_loss, dscores = layers.half_square_loss(scores, y)
        reg_loss = 0
        for f in self.params.keys():
            if f[0] == 'W':
                for w in self.params[f]:
                    reg_loss += 0.5 * self.reg * np.sum(w * w)
        loss = data_loss + reg_loss
        # ===============================
        # BACKWARD PASS
        # ===============================
        hidden['dh' + str(self.num_layers)] = dscores
        for l in range(self.num_layers)[::-1]:
            idx = l + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]

            dh, dw, db = layers.affine_backward(dh, h_cache)
            hidden['dh' + str(idx-1)] = dh
            hidden['dW' + str(idx)] = dw
            hidden['db' + str(idx)] = db

            #if idx == self.num_layers:
            #    # TODO : Make a change to loss computation here...
            #    dh, dw, db = layers.affine_backward(dh, h_cache)
            #    #rho = hidden['rho' + str(idx-1)]
            #    #dh, dw, db = auto_affine_backward(dh, h_cache, rho)
            #    hidden['dh' + str(idx-1)] = dh
            #    hidden['dW' + str(idx)] = dw
            #    hidden['db' + str(idx)] = db
            #else:
            #    # TODO: Batchnorm, etc
            #    dh, dw, db = layers.affine_relu_backward(dh, h_cache)
            #    hidden['dh' + str(idx-1)] = dh
            #    hidden['dW' + str(idx)] = dw
            #    hidden['db' + str(idx)] = db

        # Update all parameters
        dw_list = {}
        for key, val in hidden.items():
            if key[:2] == 'dW':
                dw_list[key[1:]] = val + self.reg * self.params[key[1:]]

        db_list = {}
        for key, val in hidden.items():
            if key[:2] == 'db':
                db_list[key[1:]] = val

        dgamma_list = {}
        for key, val in hidden.items():
            if key[:6] == 'dgamma':
                dgamma_list[key[1:]] = val

        dbeta_list = {}
        for key, val in hidden.items():
            if key[:5] == 'dbeta':
                dbeta_list[key[1:]] = val

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)
        grads.update(dgamma_list)
        grads.update(dbeta_list)

        return loss, grads
