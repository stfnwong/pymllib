"""
CONVNET
Some basic convolutional networks

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pymllib.layers.layers as layers
import pymllib.utils.data_utils as data_utils

# Debug
#from pudb import set_trace; set_trace()

class ConvNetLayer(object):
    """
    An L-layer convolutional network with the following architecture

    [conv-relu-pool2x2] X L - [affine - relu] x M - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 32],
                 filter_size=3, hidden_dims=[100, 100], num_classes=10,
                 weight_scale=1e-3, reg=0.0, dtype=np.float32, use_batchnorm=False,
                 verbose=False):
        """
        Init a new network
        """

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.weight_scale = weight_scale
        self.dtype = dtype
        self.bn_params = {}
        self.filter_size = filter_size
        self.L = len(num_filters)
        self.M = len(hidden_dims)

        # Internal parameter dict
        self.params = {}

        # Size of the input
        Cinput, Hinput, Winput = input_dim
        stride_conv = 1

        # Init the weights for the conv layers
        F = [Cinput] + num_filters
        for i in range(self.L):
            idx = i + 1
            W = self.weight_scale * np.random.randn(F[i+1], F[i], self.filter_size, self.filter_size)
            b = np.zeros(F[i+1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})
            # TODO: Insert batchnorm here

        # Init the weights for the affine relu layers
        # Start with the size of the last activation
        Hconv, Wconv = self._size_conv(stride_conv, self.filter_size, Hinput, Winput, self.L)
        dims = [Hconv * Wconv * F[-1]] + hidden_dims
        for i in range(self.M):
            idx = self.L + i + 1
            W = self.weight_scale * np.random.randn(dims[i], dims[i+1])
            b = np.zeros(dims[i + 1])
            self.params.update({'W' + str(idx): W,
                                'b' + str(idx): b})

        # Scoring layer
        W = self.weight_scale * np.random.randn(dims[-1], num_classes)
        b = np.zeros(num_classes)
        idx = self.L + self.M + 1
        self.params.update({'W' + str(idx): W,
                            'b' + str(idx): b})

        # Cast parameters to correct type
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def _size_conv(self, stride_conv, filter_size, H, W, n_conv):
        P = int((filter_size - 1)/ 2)
        Hc = int(1+ (H + 2 * P - filter_size) / stride_conv)
        Wc = int(1+ (W + 2 * P - filter_size) / stride_conv)
        wpool = 2
        hpool = 2
        spool = 2   # stride of pool
        Hp = int(1 + (Hc - hpool) / spool)
        Wp = int(1 + (Wc - hpool) / spool)

        if n_conv == 1:
            return Hp, Wp
        else:
            # recursively sub-divide
            H = Hp
            W = Wp
            return self._size_conv(stride_conv, filter_size, H, W, n_conv-1)

    # TODO : string

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the convnet
        """

        X = X.astype(self.dtype)
        N = X.shape[0]
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # Layer parameters
        conv_param = {'stride': 1, 'pad': int((self.filter_size - 1) / 2)}
        pool_param = {'pool_height': 2, 'pool_width': 2,  'stride': 2}

        # TODO : Batchnorm will go here
        scores = None

        blocks = {}
        blocks['h0'] = X
        # ===============================
        # FORWARD PASS
        # ===============================

        # Forward into conv block
        for l in range(self.L):
            idx = l + 1
            W = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = blocks['h' + str(idx-1)]

            # TODO: batchnorm would go here
            h, cache_h = layers.conv_relu_pool_forward(h, W, b, conv_param, pool_param)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Forward into linear blocks
        for l in range(self.M):
            idx = self.L + l + 1
            h = blocks['h' + str(idx-1)]
            if l == 0:
                h = h.reshape(N, np.prod(h.shape[1:]))

            W = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            # TODO : batchnorm would go here
            h, cache_h = layers.affine_relu_forward(h, W, b)
            blocks['h' + str(idx)] = h
            blocks['cache_h' + str(idx)] = cache_h

        # Forward into the score
        idx = self.L + self.M + 1
        W = self.params['W' + str(idx)]
        b = self.params['b' + str(idx)]
        h = blocks['h' + str(idx-1)]
        h, cache_h = layers.affine_forward(h, W, b)
        blocks['h' + str(idx)] = h
        blocks['cache_h' + str(idx)] = cache_h

        scores = blocks['h' + str(idx)]

        if y is None:
            return scores

        loss = 0.0
        grads = {}
        # Compute the loss
        data_loss, dscores = layers.softmax_loss(scores, y)
        reg_loss = 0.0
        for k in self.params.keys():
            if k[0] == 'W':
                for w in self.params[k]:
                    reg_loss += 0.5 * self.reg * np.sum(w * w)
        loss = data_loss + reg_loss

        # ===============================
        # BACKWARD PASS
        # ===============================
        idx = self.L + self.M + 1
        dh = dscores
        h_cache = blocks['cache_h' + str(idx)]
        dh, dW, db = layers.affine_backward(dh, h_cache)
        blocks['dh' + str(idx-1)] = dh
        blocks['dW' + str(idx)] = dW
        blocks['db' + str(idx)] = db

        # Backprop into the linear layers
        for l in range(self.M)[::-1]:
            idx = self.L + l + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            # TODO : batchnorm goes here
            dh, dW, db = layers.affine_relu_backward(dh, h_cache)
            blocks['dh' + str(idx-1)] = dh
            blocks['dW' + str(idx)] = dW
            blocks['db' + str(idx)] = db

        # Backprop into conv blocks
        for l in range(self.L)[::-1]:
            idx = l + 1
            dh = blocks['dh' + str(idx)]
            h_cache = blocks['cache_h' + str(idx)]
            if l == max(range(self.L)[::-1]):
                dh = dh.reshape(*blocks['h' + str(idx)].shape)
            # TODO : batchnorm goes here
            dh, dW, db = layers.conv_relu_pool_backward(dh, h_cache)
            blocks['dh' + str(idx-1)] = dh
            blocks['dW' + str(idx)] = dW
            blocks['db' + str(idx)] = db

        # Add reg term to W gradients
        dw_list = {}
        for key, val in blocks.items():
            if key[:2] == 'dW':
                dw_list[key[1:]] = val + self.reg * self.params[key[1:]]

        db_list = {}
        for key, val in blocks.items():
            if key[:2] == 'db':
                db_list[key[1:]] = val

        ## TODO : This is a hack
        #dgamma_list = {}
        #for key, val in hidden.items():
        #    if key[:6] == 'dgamma':
        #        dgamma_list[key[1:]] = val

        ## TODO : This is a hack
        #dbeta_list = {}
        #for key, val in hidden.items():
        #    if key[:5] == 'dbeta':
        #        dbeta_list[key[1:]] = val

        grads = {}
        grads.update(dw_list)
        grads.update(db_list)


        return loss, grads


class ThreeLayerConvNet(object):
    """
    A three layer convolutional network with the following architecture

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W, and with C input
    channels

    """

    # TODO : Use **kwargs
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=1.0,
                 dtype=np.float32, use_batchnorm=False, verbose=False):
        """
        TODO : Docstring
        """

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.dtype = dtype
        self.weight_scale = weight_scale
        self.bn_params = {}
        self.params = {}

        # Init weights and biases for three-layer convnet
        C, W, H = input_dim
        F = num_filters
        # Only use square filters
        Fh = filter_size
        Fw = filter_size
        stride_conv = 1     # TODO : make a settable param
        P = (filter_size - 1 ) / 2  # pad
        Hc = int(1 + (H + 2 * P - Fh) / stride_conv)
        Wc = int(1 + (W + 2 * P - Fw) / stride_conv)

        W1 = self.weight_scale * np.random.randn(F, C, Fh, Fw)
        b1 = np.zeros(F)

        # Pool layer
        # ======== DIMS ========
        # Input - (N, F, Hc, Wc)
        # Output - (N, F, Hp, Wp)

        w_pool = 2
        h_pool = 2
        stride_pool = 2
        Hp = int(1 + (Hc - h_pool) / stride_pool)
        Wp = int(1 + (Wc - w_pool) / stride_pool)

        # Hidden affine layer
        # ======== DIMS ========
        # Input - (N, Fp * Hp * Wp)
        # Output - (N, Hh)
        Hh = hidden_dim
        W2 = self.weight_scale * np.random.randn(F * Hp * Wp, Hh)
        b2 = np.zeros(Hh)

        # Output affine layer
        # ======== DIMS ========
        # Input - (N, Hh)
        # Output - (N, Hc)
        Hc = num_classes
        W3 = self.weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)

        self.params.update({
            'W1': W1,
            'W2': W2,
            'W3': W3,
            'b1': b1,
            'b2': b2,
            'b3': b3})

        # TODO : batchnorm params

        # Convert datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convnet
        """

        X = X.astype(self.dtype)  # convert datatype
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # TODO: Batchnorm here

        N = X.shape[0]
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # TODO : more batchnorm stuff here

        fsize = W1.shape[2]
        conv_param = {'stride': 1,
                      'pad': int((fsize - 1) / 2)}
        pool_param = {'pool_height': 2,
                      'pool_width': 2,
                      'stride': 2}
        scores = None

        # ===============================
        # FORWARD PASS
        # ===============================
        x = X
        w = W1
        b = b1
        # Forward into the conv layer
        # TODO : batchnorm
        conv_layer, cache_conv_layer = layers.conv_relu_pool_forward(x, w, b, conv_param, pool_param)

        N, F, Hp, Wp = conv_layer.shape     # Shape of output

        # Forward into the hidden layer
        x = conv_layer.reshape((N, F, Hp * Wp))
        w = W2
        b = b2
        hidden_layer, cache_hidden_layer = layers.affine_relu_forward(x, w, b)
        N, Hh = hidden_layer.shape

        # Forward into linear output layer
        x = hidden_layer
        w = W3
        b = b3
        scores, cache_scores = layers.affine_forward(x, w, b)

        if y is None:
            return scores

        loss = 0
        grads = {}
        # ===============================
        # BACKWARD PASS
        # ===============================
        data_loss, dscores = layers.softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss = 0.5 * self.reg * np.sum(W2**2)
        reg_loss = 0.5 * self.reg * np.sum(W3**2)
        loss = data_loss + reg_loss

        # backprop into output layer
        dx3, dW3, db3 = layers.affine_backward(dscores, cache_scores)
        dW3 += self.reg * W3

        # backprop into first fc layer
        dx2, dW2, db2 = layers.affine_relu_backward(dx3, cache_hidden_layer)
        dW2 += self.reg * W2

        # Backprop into conv layer
        dx2 = dx2.reshape(N, F, Hp, Wp)           # Note - don't forget to reshape here...
        dx, dW1, db1 = layers.conv_relu_pool_backward(dx2, cache_conv_layer)
        dW1 += self.reg * W1

        grads.update({
            'W1': dW1,
            'W2': dW2,
            'W3': dW3,
            'b1': db1,
            'b2': db2,
            'b3': db3})

        return loss, grads
