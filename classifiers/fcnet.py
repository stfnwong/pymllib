"""
FULL-CONNECTED NETWORK
A more modular design in the style of Caffe

TODO : Implement the layers as objects and produce an object oriented design
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
import numpy as np
import layers
import data_utils
import solver

# Debug
#from pudb import set_trace; set_trace()


class FCNet(object):
    """
    TODO: Docstring
    """
    def __init__(self, hidden_dims, input_dim, num_classes=10,
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

        # Initialize the parameters of the network, storing all values into a
        # dictionary at self.params. The keys to the dictionary are W1, b1 fo
        # layer 1, W2, b2 for layer 2, and so on.
        if type(hidden_dims) is not list:
            raise ValueError('hidden_dim must be a list')

        dims = [input_dim] + hidden_dims + [num_classes]
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
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def __str__(self):
        s = []
        # TODO : How do we know what the type of the layer is when the weight
        # information and the activation are separated?
        for l in range(self.num_layers):
            wl = self.params['W' + str(l+1)]
            bl = self.params['b' + str(l+1)]
            s.append('Layer %d\n\t W%d: (%d, %d),\t b%d: (%d)\n' % (l+1, l+1, wl.shape[0], wl.shape[1], l+1, bl.shape[0]))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

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
            for k, bn_param in self.bn_params.items():
                bn_param[mode] = mode

        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))   # TODO ; Check this...

        if self.use_dropout:
            hdrop, cache_hdrop = layers.layers.dropout_forward(hidden['h0'],
                                                 self.dropout_param)
            hidden['hdrop0'] = hdrop
            hidden['cache_hdrop0'] = cache_hdrop

        # Iterate over layers
        # TODO : How do we combine various (separate) layers together?
        # In the object-oriented model we just call forward on each layer,
        # and since the 'type' of the layer is determined by the object
        # we can combine layers in any order. This makes things like generating
        # a pretty print version of the network simpler, since we can just
        # iterated over the layers, produce their individual strings, and
        # arrange them into a single output
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            if self.use_batchnorm and idx != self.num_layers:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                bn_param = self.bn_params['bn_param' + str(idx)]

            # Compute the forward pass
            # output layer is a special case
            if idx == self.num_layers:
                h, cache_h = layers.affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            else:
                if self.use_batchnorm:
                    h, cache_h = layers.affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h
                else:
                    h, cache_h = layers.affine_relu_forward(h, w, b)
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
        data_loss, dscores = layers.softmax_loss(scores, y)
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
                dh, dw, db = layers.affine_backward(dh, h_cache)
                hidden['dh' + str(idx-1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db
            else:
                if self.use_dropout:
                    cache_hdrop = hidden['cache_hdrop' + str(idx)]
                    dh = layers.cdropout_backward(dh, cache_hdrop)
                if self.use_batchnorm:
                    dh, dw, db, dgamma, dbeta = layers.affine_norm_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = layers.affine_relu_backward(dh, h_cache)         # TODO This layer definition
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        # w gradients where we add the regularization term
        # TODO :' Tidy this up
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



class FCNetObject(object):
    def __init__(self, hidden_dims, input_dim, layer_types, num_classes=10,
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

        # Initialize the parameters of the network, storing all values into a
        # dictionary at self.params. The keys to the dictionary are W1, b1 fo
        # layer 1, W2, b2 for layer 2, and so on.
        if type(hidden_dims) is not list:
            raise ValueError('hidden_dim must be a list')

        if type(layer_types) is not list:
            raise ValueError("layer_types must be a list")

        if len(hidden_dims) != len(layer_types):
            print('DEBUG : len(hidden_dims) != len(layer_types)')

        dims = [input_dim] + hidden_dims + [num_classes]
        self.layers = []
        for i in range(len(dims)-1):
            if layer_types[i] == 'affine':
                l = layers.AffineLayer(dims[i], dims[i+1], weight_scale)
            elif layer_types[i] == 'relu':
                l = layers.ReLULayer(dims[i], dims[i+1], weight_scale)
            elif layer_types[i] == 'relu-affine':
                l = layers.ReluAffineLayer(dims[i], dims[i+1], weight_scale)

            self.layers.append(l)

    def __str__(self):
        s = []
        s.append("%d layer network\n" % self.num_layers)
        s.append("Hidden layers\n")
        for l in range(len(self.layers)):
            s.append('Layer %d : %s\n' % (l+1, self.layers[l]))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()


    def collect_params(self):
        """
        Collect params.
        This is just a compatability layer for the solver. In a real object oriented
        design the solver will need to be re-written
        """

        self.params = {}

        for i in range(len(self.layers)):
            self.params['W' + str(i+1)] = self.layers[l].W
            # TODO: Where to the graients get stored in this configuration?

    def loss(self, X, y=None):
        print('TODO')
