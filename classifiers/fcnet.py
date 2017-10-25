"""
FULL-CONNECTED NETWORK
A more modular design in the style of Caffeo


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

class TwoLayerNet(object):
    def __init__(self, input_dim=(32*32*3), hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, verbose=False):
        """
        Init a new two layer network.
        This is just to ensure the unit test works correctly,
        don't keep this module
        """
        self.params = {}
        self.reg = reg
        self.D = input_dim
        if type(hidden_dim) is list:
            self.M = hidden_dim[0]
        else:
            self.M = hidden_dim
        self.C = num_classes

        w1 = weight_scale * np.random.randn(self.D, self.M)
        w2 = weight_scale * np.random.randn(self.M, self.C)
        b1 = np.zeros(self.M)
        b2 = np.zeros(self.C)

        self.params.update({'W1': w1,
                            'W2': w2,
                            'b1': b1,
                            'b2': b2})

    def loss(self, X, y=None):
        """
        Compute loss and gradient
        """

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        X = X.reshape(X.shape[0], self.D)
        # Forward pass
        hidden_layer, cache_hidden_layer = layers.affine_relu_forward(X, W1, b1)
        scores, cache_scores = layers.affine_forward(hidden_layer, W2, b2)

        # Return if we are in training mode
        if y is None:
            return scores

        data_loss, dscores = layers.softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss

        # backward pass
        grads = {}
        # Second layer
        dx1, dW2, db2 = layers.affine_backward(dscores, cache_scores)
        dW2 += self.reg * W2
        # First layer
        dx, dW1, db1 = layers.affine_relu_backward(dx1, cache_hidden_layer)
        dW1 += self.reg * W1

        grads.update({'W1': dW1,
                      'W2': dW2,
                      'b1': db1,
                      'b2': db2})

        return loss, grads



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
            s.append('Layer %d\n\t W: (%d, %d), b: (%d)\n' % (l+1, wl.shape[0], wl.shape[1], bl.shape[0]))

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
                bn_param = self.bn_params['bn_paramm' + str(idx)]

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
        hidden['dh' + str(self.num_layers)] = dscores
        for l in range(self.num_layers)[::-1]:
            idx = l + 1
            dh = hidden['dh' + str(idx)]
            h_cache = hidden['cache_h' + str(idx)]

            if idx == self.num_layers:
                dh, dw, db = layers.layers.affine_backward(dh, h_cache)
                hidden['dh' + str(idx-1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db
            else:
                if self.use_dropout:
                    cache_hdrop = hidden['cache_hdrop' + str(idx)]
                    dh = layers.layers.cdropout_backward(dh, cache_hdrop)
                if self.use_batchnorm:
                    dh, dw, db, dgamma, dbeta = layers.layers.affine_norm_relu_backward(dh, h_cache)
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db
                    hidden['dgamma' + str(idx)] = dgamma
                    hidden['dbeta' + str(idx)] = dbeta
                else:
                    dh, dw, db = layers.layers.affine_relu_backward(dh, h_cache)         # TODO This layer definition
                    hidden['dh' + str(idx-1)] = dh
                    hidden['dW' + str(idx)] = dw
                    hidden['db' + str(idx)] = db

        # w gradients where we add the regularization term
        # TODO :' Tidy this up
        for key, val in hidden.items():
            if key[:2] == 'dW':
                dw_list = {key[1:]: val + self.reg * self.params[key[1:]]}

        for key, val in hidden.items():
            if key[:2] == 'db':
                db_list = {key[1:]: val}

        for key, val in hidden.items():
            if key[:6] == 'dgamma':
                dgamma_list = {key[1:]: val}

        for key, val in hidden.items():
            if key[:5] == 'dbeta':
                dbeta_list = {key[1:]: val}

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)
        grads.update(dgamma_list)
        grads.update(dbeta_list)

        return loss, grads


# TODO : This should eventually be elsewhere
def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# ======== SOME BASIC TEST CODE ======== #
if __name__ == "__main__":

    # Get some data
    data_dir = 'datasets/cifar-10-batches-py'
    dataset = data_utils.get_CIFAR10_data(data_dir)
    for k, v in dataset.items():
        print("%s : %s" % (k, v.shape))


    hidden_dims = [200]
    input_dim = 32 * 32 * 3
    fcnet = FCNet(hidden_dims, input_dim)

    s = solver.Solver(fcnet, data, update_rule='sgd',
                      optim_config={'learning_rate': 1e-3},
                      lr_decay = 0.95,
                      num_epochs=2,
                      batch_size=250,
                      print_every=100)
