"""
FULL-CONNECTED NETWORK
A more modular design in the style of Caffeo


"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
import numpy as np
import layers


class FCNet(object):
    def __init__(self, hidden_dims, input_dim, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float332, seed=None,
                 verbose=False):

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network, storing all values into a
        # dictionary at self.params. The keys to the dictionary are W1, b1 fo
        # layer 1, W2, b2 for layer 2, and so on.
        if type(hidden_dims) is not list:
            raise ValueError('hidden_dim must be a list')

        dims = [input_dim] + hidden_dims + [num_classes]
        # TODO ; Do I need to save these parameters for later....?
        # Weight dict
        Ws = {'W' + str(i+1) : weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims)-1)}
        bs = {'b' + str(i+1) : np.zeros(dims[i+1]) for i in range(len(dims)-1)}

        self.params.update(bs)
        self.params.update(Ws)

        # When using dropout, we must pass a dropout_param dict to each dropout
        # layer so that the layer knows the dropout probability and the mode
        # (train/test).
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode' : 'train', 'p' : dropout}
            if self.verbose:
                print("Using dropout with p = %f" % (self.dropout_param['p']))
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. We pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the
        # forward pass of the second batch norm layer, and so on
        if self.use_batchnorm:
            self.bn_params = {'bn_params' + str(i+1) : {'mode' : 'train',
                                                        'running_mean' : np.zeros(dims[i+1]),
                                                        'running_var'  : np.zeros(dims[i+1])}
                              for i in range(len(dims)-2) }
            gammas = {'gamma' + str(i+1) : np.ones(dims[i+1])
                      for i in range(len(dims)-2)}
            betas = {'beta' + str(i+1) : np.zeros(dims[i+1])
                     for i in range(len(dims)-2)}
            self.params.update(gammas)
            self.params.update(betas)

        # Cast params to correct data type
        if(self.verbose):
            print("Casting parameters to type %s" % (self.dtype))
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    # TODO : String functions

    def loss(self, X, y=None):
        """
        LOSS

        Compute loss and gradients for the fully connected network
        """

        X = X.astype(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # Set batchnorm params based on whether this is a training or a test
        # run
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for k, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))   # TODO ; Check this...

        if self.use_dropout:

            hdrop, cache_hdrop = dropout_forward(hidden['h0'],
                                                 self.dropout_param)
            hidden['hdrop0'] = hdrop
            hidden['cache_hdrop0'] = cache_hdrop

        # Iterate over layers
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            if self.use_batchnorm:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                bn_param = self.bn_params['bn_paramm' + str(idx)]

            # Compute the forward pass
            # output layer is a special case
            if idx == self.num_layers:
                # TODO ; re-create the layer definitions
                h, cache_h = layers.affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            else:
                if self.use_batchnorm:
                    h, cache_h = layers.affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                else:
                    h, cache_h = layers.affine_relu_forward(h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h

                if self.use_dropout:
                    h = hidden['h' + str(idx)]
                    hdrop, cache_hdrop = dropout_forward(h, self.dropout_param)
                    hidden['hdrop' + str(idx)] = hdrop
                    hidden['cache_hdrop' + str(idx)] = cache_hdrop

        scores = hidden['h' + str(self.num_layers)]

        if mode == 'test':
            return scores

        loss = 0.0
        grads = {}
        # Compute loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0
        for f in self.params.keys():
            if f[0] == 'W':
                for w in self.params[f]:
                    reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss
        # ===============================
        # BACKWARD PASS
        # ===============================
        hidde['dh' + str(self.num_layers)] = dscores
        for l in range(self.num_layers)[::-1]:
            idx = i + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]

            if idx == self.num_layers:
                dh, hw. db = affine_backwards(dh, h_cache)
                hidden['dh' + str(idx-1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db
            else:
                if self.use_dropout:
                    cache_hdrop = hidden['cache_hdrop' + str(idx)]
                    dh = dropout_backward(dh, cache_hdrop)
                if self.use_batchnorm:
                    dh, dw, db, dgamma, dbeta = layers.layers.affine_norm_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = affine_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        # w gradients where we add the regularization term
        # TODO :' Tidy this up
        for key, val in hidden.iteritems():
            if key[:2] == 'dW':
                dw_list = {key[1:]: val + self.reg * self.params[key[1:]]}

        for key, val in hidden.iteritems():
            if key[:2] == 'db':
                db_list = {key[1:]: val}

        for key, val in hidden.iteritems():
            if key[:6] == 'dgamma':
                dgamma_list = {key[1:]: val}

        for key, val in hidden.iteritems():
            if key[:5] == 'dbeta':
                dbeta_list = {key[1:]: val}

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)
        grads.update(dgamma_list)
        grads.update(dbeta_list)

        return loss, grads






# ======== SOME BASIC TEST CODE ======== #
if __name__ == "__main__":

    hidden_dims = [200]
    input_dim = 32 * 32 * 3
    fcnet = FCNet(hidden_dims, input_dim)
