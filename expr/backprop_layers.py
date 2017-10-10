"""
Backprop with layer objects

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import numpy as np
import neural_net_utils as nnu

# Debug
from pudb import set_trace; set_trace()


class Layer(object):
    def __init__(self, input_dim, layer_size, layer_sd=0.01):
        self.layer_sd = layer_sd
        self.W = self.layer_sd * np.random.randn(input_dim, layer_size)
        self.b = np.zeros((1, layer_size))

    def update(self, dW, db, step_size):
        self.W += (-step_size) * dW
        self.b += (-step_size) * db

# Linear layer
class LinearLayer(Layer):
    def __str__(self):
        s = []
        s.append('ReLU Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__

    def forward(self, X):
        return np.dot(X, self.W) + self.b

    def backward(self, dz, X_prev):
        return np.dot(X_prev.T, dz)

# Layer with ReLU activation
class ReLULayer(Layer):
    def __str__(self):
        s = []
        s.append('ReLU Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__

    def forward(self, X):
        Z =  np.dot(X, self.W) + self.b
        return np.maximum(0, Z)

    def backward(self, dz, X_prev):
        d = np.zeros_like(dz)
        d[X_prev > 0] = 1
        return dz * d

# Layer with Sigmoid Activation
class SigmoidLayer(Layer):
    def __str__(self):
        s = []
        s.append('Sigmoid Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, dz, X_prev):  # Here to keep function prototype symmetry
        return dz * (1 - dz)



# ==== NETWORK FUNCTIONS ==== #

def main():
    # Generate some data
    N = 500
    h = 100
    D = 2
    K = 3
    theta = 0.3
    spiral_data = nnu.create_spiral_data(N, D, K, theta)
    X = spiral_data[0]      # data
    y = spiral_data[1]      # labels

    num_examples = X.shape[0]

    # Create layers
    layer_sizes = [h, 3]
    relu_hidden = ReLULayer(D, layer_sizes[0])
    linear_output = LinearLayer(layer_sizes[1], K)
    layer_list = [relu_hidden, linear_output]

    # Gradient descent, Forward Pass
    layer_input = X
    Z_list = []
    for l in layer_list:
        layer_out = l.forward(layer_input)  # extra var for watching in debugger
        layer_input = layer_out
        Z_list.append(layer_out)        # for later inspection

    # Compute scores, loss
    exp_scores = np.exp(layer_out) # same as np.exp(Z_list[-1])
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(logprobs) / num_examples
    reg_loss = 0   # TODO
    loss = data_loss + reg_loss

    # Compute gradient on loss
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # Gradient descent, backward pass
    dz_list = []
    dz_prev = dscores
    for l in range(len(layer_list)):
        idx = len(layer_list) - l - 1
        # Z[idx] is the activation on layer idx
        dz = layer_list[l].backward(dz_prev, Z_list[idx])
        dz_prev = dz
        dz_list.append(dz)




if __name__ == "__main__":
    main()
