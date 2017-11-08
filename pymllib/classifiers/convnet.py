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

        # TODO: batchnorm stuff
        for k, v in grads.items():
            print("%s : max = %f, min = %f" % (k, np.max(v), np.min(v)))

        return loss, grads




#class ConvNet(object):



