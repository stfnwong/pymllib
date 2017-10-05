"""
BACKPROP
Just an implementation of the backprop mechanics.

Stefan Wong 2017
"""

import numpy as np
import neural_net_utils as nnu

# Debug
#from pudb import set_trace; set_trace()

# Some simple activation functions here
def relu(z):
    return np.maximum(0, z)

def relu_dz(dz, X):
    d = np.zeros_like(dz)
    d[X > 0] = 1
    return d * dz


N = 500
h = 100
D = 2
K = 3
theta = 0.3
spiral_data = nnu.create_spiral_data(N, D, K, theta)
# Get some data
X = spiral_data[0]
y = spiral_data[1]

# Network properties
input_dim = 2
#layer_dims = [100, 3]
layer_dims = [100, 100, 3]
num_layers = len(layer_dims) + 1

# Internals are global for now


def init_weights(input_dim, layer_dims):
    W = []
    B = []
    for n in range(num_layers-1):
        if(n == 0):
            w = 0.01 * np.random.randn(input_dim, layer_dims[n])
        else:
            w = 0.01 * np.random.randn(layer_dims[n-1], layer_dims[n])
        b = np.zeros((1, layer_dims[n]))
        W.append(w)
        B.append(b)

    return W, B

# Pre-allocate memory for weights, biases, and activations
W, B = init_weights(input_dim, layer_dims)
Z = [np.zeros(weight.shape) for weight in W]

# Pre-allocate memory for derivatives
dW = []
dB = []
for n in range(num_layers-1):
    dw = np.zeros((input_dim, layer_dims[n]))
    db = np.zeros((1,1))
    dW.append(dw)
    dB.append(db)


def backprop(X, y, f, df, reg=1e-1, step_size=1e-3):
    # Forward pass
    layer_input = X
    for n in range(len(W)):
        activation = np.dot(layer_input, W[n]) + B[n]
        Z[n] = f(activation)
        layer_input = Z[n]

    # TODO : Compute loss
    loss = 0

    #dW[len(dW)-1] =
    for n in range(len(Z)):
        idx = len(Z) - n - 1
        #dZ = df()
        dx = np.dot(W[idx].T, df(Z[idx], X))
        dW[idx] = np.dot(dW[idx].T, dx)
        dB[idx] = np.sum(dW[idx], axis=0, keepdims=True)

    return dW, dB, loss

# TODO : How to make this more general?
def backprop_sigmoid(X, y, reg=1e-1, step_size=1e-3):
    # Forward pass
    layer_input = X
    print("Forward pass")
    for n in range(len(W)):
        x = np.dot(layer_input, W[n]) + B[n]
        Z[n] = 1 / (1 + -np.exp(x))
        layer_input = Z[n]
        print("Layer %d : activation shape is (%d,%d)" % (n+1, x.shape[0], x.shape[1]))

    print("\n")
    loss = 0

    # Backward pass
    dZ = []
    dW = []
    dB = []
    print("Backward pass")
    for n in range(len(Z)):
        idx = len(Z) - n - 1
        dz = Z[idx] * (1 - Z[idx])
        #dx = np.dot(W[idx].T, dz)
        dw = np.outer(dz, X)
        print("Layer %d : dz.shape is (%d, %d)" % (idx, dz.shape[0], dz.shape[1]))
        dW.append(dw)
        dZ.append(dz)

    return dW, dB, loss


def main():
    #dW, db, loss = backprop(X, y, relu, relu_dz)
    dW, db, loss = backprop_sigmoid(X, y)

if __name__ == "__main__":
    main()
