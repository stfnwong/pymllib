"""
Train a 2-layer network on spiral data

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import numpy as np
import matplotlib.pyplot as plt
import ref_classifiers as rc
# Internal libs
import neural_net_utils as nnu
# TODO ; Forget about external activations for now
#import activations as activ
#Debug
from pudb import set_trace; set_trace()

# ReLU functions
def relu_forward(W, X, y, b):
    return np.maximum(0, X)

def relu_backward(dz, X):
    d = np.zeros_like(dz)
    d[X > 0] = 1
    dx = d * dz

    return dx

# Sigmoid functions
def sigmoid_forward(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_backward(z):
    return sigmoid(z) * (1-sigmoid(z))

"""
A generic layer class

I think that it may be better to walk through the layers as if they
are nodes in a graph
"""
class NNLayer(object):
    def __init__(self):
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.scores = None
        #self.f = None

    def init_weights(self, D, lsize):
        self.W = np.random.randn(D, lsize)
        self.b = np.zeros((1,lsize))

    def forward(self, f, X, y):
        return f(self.W, X, y, self.b)

    def backward(self, f, dz, X):

        return f(dz, X)

    # Update internal weights
    def update(self, ss, dW, db):
        self.W += (-ss) * dW
        self.b += (-ss) * db

    def get_size(self):
        return np.sum(W.shape)


"""
Neural Network that implements an arbitrary number of hidden layers
"""
class ModularNetwork(object):
    def __init__(self, layer_dims, layer_sizes, input_dim, reg=1e-3):
        # NOTE: input_dim should be a single integer value representing the
        # total size of the input
        self.reg = reg
        # Hidden layers
        self.layer_dims = layer_dims
        self.layer_sizes = layer_sizes
        self.num_layers = 1 + len(layer_dims)
        self.layers = []

        # Init layers
        for n in range(self.num_layers-1):
            l = NNLayer()
            l.init_weights(layer_dims[n], layer_sizes[n])
            self.layers.append(l)
            if(n == self.num_layers):
                bgrad = np.sum(dscores)

        # Storing loss, etc
        self.store_loss = False

    # Return a large array containing all the weights of all the layers
    def get_weights(self):
        Wt = np.zeros((self.num_layers-1, np.max(self.layer_dims)))
        # TODO : Fill Wt
        return Wt

    def compute_scores(self, X, y):
        num_examples = X.shape[0]
        lscores = X
        # Forward pass over layers
        for n in range(self.num_layers-1):
            lscores = self.layers[n].forward(relu_forward, lscores, y)
            self.layers[n].scores = lscores

        return lscores

    def backprop(self, dscores):

        prev_grad = dscores
        # Backward pass over layers
        for n in range(self.num_layers, 2, -1):
            lgrad = self.layers[n].backward(relu_backward, prev_grad)
            bgrad = np.sum(prev_grad, axis=0, keepdims=True)

    """
    Compute the loss from the scores
    """
    def compute_loss(self, probs, labels, num_examples):

        logprobs = -np.log(probs[range(num_examples), labels])
        data_loss = np.sum(logprobs) / num_examples
        # Compute the regularization loss
        reg_loss = 0
        for n in range(self.num_layers-1):
            reg_loss += 0.5 * self.reg * np.sum(self.layers[n].W * self.layers[n].W)
        total_loss = data_loss + reg_loss

        return total_loss

    def train(self, X, y):

        num_examples = X.shape[0]

        # TODO : Iteration goes here
        scores = self.compute_scores(X, y)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Forward pass
        loss = self.compute_loss(probs, y, num_examples)
        # Backward pass
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores = dscores / num_examples
        grads = self.backprop(scores)




# ======== ENTRY POINT ======== #
if __name__ == "__main__":

    N = 500
    h = 100
    D = 2
    K = 3
    theta = 0.3
    spiral_data = nnu.create_spiral_data(N, D, K, theta)

    ref_net = rc.TwoLayerNetwork()
    #net.init_params(h, D, K)
    num_iters = 10000

    X = spiral_data[0]      # data
    y = spiral_data[1]      # labels

    #NNFunction(X, y)
    W1, W2, b1, b2 = ref_net.train(X, y, num_iters)


    #for n in range(num_iters):
    #    loss, grads = net.grad_descent(X, y)
    #    if(n % 1000 == 0):
    #        print("Iter %d, loss = %f" % (n, loss))
    #loss, grads = net.grad_descent(X, y)

    # Visualize the classifier
    h = 0.02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    #W = (net.W1, net.W2)
    #b = (net.b1, net.b2)
    W = (W1, W2)
    b = (b1, b2)
    if(type(W) is tuple):
        Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W[0]) + b[0]), W[1]) + b[1]
    else:
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    #self.fig_classifier = plt.figure()
    plt.figure(1)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
