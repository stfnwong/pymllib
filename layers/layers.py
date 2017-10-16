"""
LAYERS

Stefan Wong 2017
"""

import numpy as np
# Debug
#from pudb import set_trace; set_trace()

# TODO : Implement mini-batch support....
"""
LAYER

The generic layer object. This holds members common to all layer
types.

"""
class Layer(object):
    def __init__(self, input_dim, layer_size, layer_sd=0.01):
        self.layer_sd = layer_sd
        self.W = self.layer_sd * np.random.randn(input_dim, layer_size)
        self.b = np.zeros((1, layer_size))
        self.Z = None       # TODO : Pre-allocate...

    def update(self, dW, db, step_size):
        self.W += (-step_size) * dW
        self.b += (-step_size) * db

# Linear layer
class AffineLayer(Layer):
    def __str__(self):
        s = []
        s.append('Linear Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' %
                 (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def forward(self, X):
        self.Z = np.dot(X, self.W) + self.b
        return self.Z

    def backward(self, dz, X_prev):
        # TODO : calculate all gradients here
        return np.dot(X_prev.T, dz)

# Layer with ReLU activation
class ReLULayer(Layer):
    def __str__(self):
        s = []
        s.append('ReLU Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' %
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
        s.append('Sigmoid Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' %
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


# TODO ; Batchnorm layer
