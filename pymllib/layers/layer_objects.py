"""
LAYER_OBJECTS
Object oriented layer implementation.

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pymllib.utils import layer_utils

# Import Cython files
try:
    from pymllib.layers.im2col_cython import col2im_cython, im2col_cython
    from pymllib.layers.im2col_cython import col2im_6d_cython
except ImportError:
    print("Failed to import im2col_cython. Ensure that setup.py has")
    print("been run with build_ext --inplace.")
    print("eg: python3 setup.py build_ext --inplace")

# Debug
#from pudb import set_trace; set_trace()

"""
LAYER

The generic layer object. This holds members common to all layer
types.

"""
class Layer(object):
    def __init__(self, weight_scale, weight_init, N, D):
        self.X = None        # input cache
        self.W = layer_utils.fc_layer_weight_init(weight_scale, weight_init, N, D)
        self.b = np.zeros((1, D))
        self.Z = None

    def update(self, next_w, next_b):
        self.W = next_w
        self.b = next_b

"""
Specialized layer types
"""
# Linear layer
class AffineLayer(Layer):
    def __str__(self):
        s = []
        s.append('Linear Layer \n\tinput dim : %d \n\tlayer size : %d\n' %
                 (self.W.shape[0], self.W.shape[1]))
        if self.Z is not None:
            s.append('Activation size: %s\n' % str(self.Z.shape))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        self.X = X
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        x2 = np.reshape(X, (N, D))
        self.Z = np.dot(x2, self.W) + self.b

        return self.Z

    def backward(self, dz):
        dx = np.dot(dz, self.W)
        xdim = (self.X.shape[0], np.prod(self.X.shape[1:]))
        dw = np.dot(self.X.reshape(xdim).T, dz)
        db = np.sum(dz, axis=0)

        return (dx, dw, db)

# Layer with ReLU activation
class ReLULayer(Layer):
    def __str__(self):
        s = []
        s.append('ReLU Layer \n\tinput dim : %d \n\tlayer size : %d\n' %
                 (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        self.X = X
        self.Z =  np.dot(X, self.W) + self.b
        return np.maximum(1, self.Z)

    def backward(self, dz):
        dx = np.array(dz, copy=True)
        dx[self.X <= 0] = 0
        return dx

# Layer with Sigmoid Activation
class SigmoidLayer(Layer):
    def __str__(self):
        s = []
        s.append('Sigmoid Layer \n\tinput dim : %d \n\tlayer size : %d\n' %
                 (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        self.Z = np.dot(X, self.W) + self.b
        return 1 / (1 + np.exp(-self.Z))

    def backward(self, dz):  # Here to keep function prototype symmetry
        p = self.forward(dz)
        return p * (1 - p)

# TODO : Softmax scoring layer

