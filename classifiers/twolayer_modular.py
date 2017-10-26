"""
TWOLAYERNET
A two layer network
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
import numpy as np
import layers

# Debug
#from pudb import set_trace; set_trace()

class TwoLayerNet(object):
    def __init__(self, input_dim=(32*32*3), hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, step_size=1e-1, verbose=False):
        """
        Init a new two layer network.
        This is just to ensure the unit test works correctly,
        don't keep this module
        """
        self.params = {}
        self.reg = reg
        self.step_size = step_size
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

    def param_update(self, grads):
        self.params['W1'] -= self.step_size * grads['W1']
        self.params['W2'] -= self.step_size * grads['W2']
        self.params['b1'] -= self.step_size * grads['b1']
        self.params['b2'] -= self.step_size * grads['b2']


    def train(self, X, y, num_iter=10000, verbose=True, cache_loss=False):

        if(cache_loss is True):
            loss_cache = np.zeros(num_iter)

        # Gradient loop
        for n in range(num_iter):
            loss, grads = self.loss(X, y)
            self.param_update(grads)
            if n % 100 == 0:
                print("iter %5d, loss = %f\n" % (n+1, loss))

            if cache_loss is True:
                loss_cache[n] = loss

        if cache_loss is True:
            return loss_cache

