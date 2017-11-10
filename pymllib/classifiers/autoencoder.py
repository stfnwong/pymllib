"""
AUTOENCODER
Attempt to use an fcnet as an autoencoder

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
import numpy as np
import layers
import data_utils

# Debug
from pudb import set_trace; set_trace()


# TODO : Write the loss function for sparse autoencoder
def half_square_loss(X, y):

    probs = np.exp(X - np.max(X, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = X.shape[0]

    loss = np.sum(np.abs(X - y), axis=1, keepdims=True)
    loss = (0.5 * (s**2)) / N

    # activations (a) are X here
    dx = 0      # shut linter up
    p = probs.copy()
    #np.sum(p - y) *

    return loss, dx


def auto_loss(X, y, rho, beta=1.0, sparsity=0.05):

    N = X.shape[0]
    eps = 1e-8
    probs = np.exp(X - np.max(X, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    #loss = np.sum(np.abs(X - y), axis=1, keepdims=True)
    loss = np.sum(np.abs(X - y))
    loss /= N

    # Sparsity penalty
    s1 = sparsity
    s2 = (1.0 - sparsity)
    p_loss = np.sum(s1 * np.log(s1 / rho) + s2 * np.log(s2 / (1 - rho)), axis=0)
    #p_loss = np.sum(s1 * np.log(s1 / rho) + s2 * np.log(s2 / (1 - rho)), axis=0, keepdims=True)
    # final loss
    loss += beta * p_loss
    dx = np.abs(X - y)


    return loss, dx

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

#def sparse_autoencoder_loss(X, y, beta, rho):
#
#    hs_loss, hs_dx = half_square_loss(X, y)
#

class Autoencoder(object):
    def __init__(self, hidden_dims, input_dim,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 verbose=False):

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Init the params of the network into the dictionary self.params
        dims = [input_dim] + hidden_dims + [input_dim]
        Ws = {'W' + str(i+1): weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims)-1)}
        bs = {'b' + str(i+1): np.zeros(dims[i+1]) for i in range(len(dims)-1)}
        self.params.update(bs)
        self.params.update(Ws)

        # Cast params to correct data type
        if self.verbose:
            print("Casting parameers to type %s" % self.dtype)
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

    def loss(self, X, y=None):

        X = X.astype(self.dtype)

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

        # Do an initial pass to compute rho.
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(idx-1)]

            h, cache_h = layers.affine_relu_forward(h, w, b)
            rho = (1 / h.shape[0]) * np.sum(h, axis=0, keepdims=True)
            hidden['rho' + str(idx-1)] = rho
            hidden['h' + str(idx)] = h        # This data will be discarded in the 'true' forward pass

        # Forward pass with activations
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            # Compute the forward pass
            # output layer is a special case
            if idx == self.num_layers:
                h, cache_h = layers.affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            else:
                h, cache_h = layers.affine_relu_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h


        scores = hidden['h' + str(self.num_layers)]

        if mode == 'test':
            return scores

        loss = 0.0
        grads = {}
        # Compute loss
        # Here we don't want to use the softmax loss, rather
        #data_loss, dscores = sparse_autoencoder_loss(scores, y)
        rho = hidden['rho' + str(self.num_layers-1)]
        data_loss, dscores = auto_loss(scores, y, rho)                # TODO: <- Add rho here...
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

            if idx == self.num_layers:
                # TODO : Make a change to loss computation here...
                #dh, dw, db = layers.affine_backward(dh, h_cache)
                rho = hidden['rho' + str(idx-1)]
                dh, dw, db = auto_affine_backward(dh, h_cache, rho)
                hidden['dh' + str(idx-1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db
            else:
                # TODO: Batchnorm, etc
                dh, dw, db = layers.affine_relu_backward(dh, h_cache)         # TODO This layer definition
                hidden['dh' + str(idx-1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

        # Update all parameters
        dw_list = {}
        for key, val in hidden.items():
            if key[:2] == 'dW':
                dw_list[key[1:]] = val + self.reg * self.params[key[1:]]

        db_list = {}
        for key, val in hidden.items():
            if key[:2] == 'db':
                db_list[key[1:]] = val

        # TODO : This is a hack
        dgamma_list = {}
        for key, val in hidden.items():
            if key[:6] == 'dgamma':
                dgamma_list[key[1:]] = val

        # TODO : This is a hack
        dbeta_list = {}
        for key, val in hidden.items():
            if key[:5] == 'dbeta':
                dbeta_list[key[1:]] = val

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)
        grads.update(dgamma_list)
        grads.update(dbeta_list)

        #if dgamma_list is not None:
        #    grads.update(dgamma_list)
        #if dbeta_list is not None:
        #    grads.update(dbeta_list)

        return loss, grads
