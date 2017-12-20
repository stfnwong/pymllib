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
        self.W = layer_utils.fc_layer_weight_init(weight_scale, weight_init, N, D)
        self.b = np.zeros((1, D))
        self.Z = None     # TODO : We need to know the batch size for this to work....

    def update(self, next_w, next_b):
        self.W = next_w
        self.b = next_b
"""
N = X.shape[0]
D = np.prod(X.shape[1:])
x2 = np.reshape(X, (N,D))
out = np.dot(x2, w) + b
cache = (X, w, b)
"""

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
            s.append('Activation size: %s\n' % self.Z.shape)

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        N = X.shape[0]
        D = np.prod(X.shape[1:])
        self.Z = np.dot(X.reshape(N, D), self.W) + self.b
        return self.Z

    def backward(self, dz, X):
        dx = np.dot(dz, self.W.T)
        p = np.prod(X.T, dz)
        dw = np.dot(X, p)
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
        self.Z =  np.dot(X, self.W) + self.b
        return np.maximum(0, self.Z)

    def backward(self, dz):
        d = np.zeros_like(dz)
        d[dz > 0] = 1
        return dz * d

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

