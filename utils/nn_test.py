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
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_dz(z):
    return sigmoid(z) * (1-sigmoid(z))


"""
A slightly less modular neural network
NOTE: It would be safe to assume that there is a non-trivial performance boost
when using NumPy arrays vs using lists of NumPy arrays, which is something to
consider during design
"""
class LessModularNetwork(object):
    # TODO ; input dims?
    def __init__(self, layer_dims, layer_sizes, num_iter=10000):
        self.num_layers = len(layer_dims) + 1
        self.biases = []
        self.weights = []
        self.num_iter = num_iter
        # set up weights at start of training
        self.init_weights(layer_dims, layer_sizes)

        # Debug
        self.verbose = False
        self.cache_loss = False

    def init_weights(self, layer_dims, layer_sizes):

        for d, s in zip(layer_dims, layer_sizes):
            W = np.random.randn(d, s)
            b = np.zeros((1,s))
            self.weights.append(W)
            self.biases.append(b)

    def backprop(self, X, y, f, df, reg=1e-0):

        if f is None:
            f = sigmoid
        if df is None:
            df = sigmoid_dz

        db = [np.zeros(b.shape) for b in self.biases]
        dW = [np.zeros(w.shape) for w in self.weights]

        zs = []
        scores = [X]
        activ = X
        # Forward pass
        for W, b in zip(self.weights, self.biases):
            z = np.dot(activ, W) + b
            zs.append(z)
            activ = f(z)
            scores.append(activ)

        # compute the loss for plotting
        probs = scores[-1]
        num_examples = X.shape[0]
        logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(logprobs) / num_examples
        # Compute the regularization loss
        reg_loss = 0
        for n in range(self.num_layers-1):
            reg_loss += 0.5 * reg * np.sum(self.weights[n] * self.weights[n])
        loss = data_loss + reg_loss


        # Backward pass
        final_score = scores[-1]
        final_z = zs[-1]
        delta = (final_score.T - y) * df(final_z).transpose()
        dW[-1] = delta
        db[-1] = np.dot(final_score, delta)

        for n in range(self.num_layers-1, 2, -1):

            if(self.verbose):
                print("Backward pass in layer %d of %d" % (n, self.num_layers-1))

            z = zs[n]
            dz = df(z)
            delta = np.dot(delta, self.weights[n+1]) * dz
            # add regularization to gradient
            db[n] = delta * reg
            dW[n] = np.dot(scores[n-1], delta) * reg

        return dW, db, loss

    def update(self, dW, db, step_size):
        # Update W
        for k in range(len(self.weights)):
            self.weights[k] += (-step_size) * dW[k]
        # Update b
        for k in range(len(self.biases)):
            self.biases[k] += (-step_size) * db[k]

    def train(self, X, y, step_size=1e-3, reg=1e-1):

        if(self.cache_loss):
            loss_cache = np.zeros((1, self.num_iter))

        for n in range(self.num_iter):
            dW, db, loss = self.backprop(X, y, sigmoid, sigmoid_dz, reg)
            self.update(dW, db, step_size)

            if(self.verbose):
                if(n % 100 == 0):
                    print("iter %d : loss %f" % (n, loss))

            if(self.cache_loss):
                loss_cache[n] = loss

        if(self.cache_loss):
            return self.weights, self.biases, loss_cache

        return self.weights, self.biases








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
        self.z = np.zeros(self.b.shape)

    def update(self, dW, db, ss):
        self.W += (-ss) * dW
        self.b += (-ss) * db

    def get_size(self):
        return np.sum(W.shape)


"""
Neural Network that implements an arbitrary number of hidden layers
"""
class ModularNetwork(object):
    def __init__(self, layer_dims, layer_sizes, input_dim, reg=1e-3, step_size=1e-1):
        # NOTE: input_dim should be a single integer value representing the
        # total size of the input
        self.reg = reg
        self.step_size = step_size
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

    def compute_scores(self, f, X):

        lscores = []
        zvec = []
        activ = X

        for n in range(self.num_layers-1):
            z = np.dot(activ, self.layers[n].W) + self.layers[n].b
            zvec.append(z)
            activ = f(z)
            self.layers[n].scores = activ
            lscores.append(activ)
            #lscores = self.layers[n].forward(relu_forward, lscores, y)
            #self.layers[n].scores = lscores

        return lscores, zvec

    def backprop(self, df, dscores, zvec):

        delta = dscores
        db = []
        dW = []

        for l in self.layers:
            db.append(np.zeros(l.b.shape))
            dW.append(np.zeros(l.W.shape))

        k = 0
        # Backward pass over layers
        for n in range(self.num_layers-2, 1, -1):
            dz = df(zvec[n])
            delta = np.dot(self.layers[n+1].W.T, delta) * dz
            db[k] = delta
            dW[k] = np.dot(delta, zvec[n-1].T)
            k += 1
            #lgrad = self.layers[n].backward(relu_backward, prev_grad)
            #bgrad = np.sum(prev_grad, axis=0, keepdims=True)

        return (dW, db)

    """
    Compute the loss from the scores
    """
    def compute_loss(self, probs, labels, num_examples):

        # TODO ; Check the sizes here
        logprobs = -np.log(probs[0][range(num_examples), labels])
        data_loss = np.sum(logprobs) / num_examples
        # Compute the regularization loss
        reg_loss = 0
        for n in range(self.num_layers-1):
            reg_loss += 0.5 * self.reg * np.sum(self.layers[n].W * self.layers[n].W)
        total_loss = data_loss + reg_loss

        return total_loss

    def train(self, X, y, num_iter=10000):

        num_examples = X.shape[0]
        f = sigmoid
        df = sigmoid_dz

        for i in range(num_iter):
            # Forward pass
            scores, zvec = self.compute_scores(f, X)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            loss = self.compute_loss(probs, y, num_examples)

            # debug print
            if i % 100 == 0:
                print("iter %d, loss = %f" % (i, loss))
            # Backward pass
            #delta = self.layers[self.num_layers-1].scores - y * df(zvec[self.num_layers-1])
            dscores = probs
            dscores[0][range(num_examples),y] -= 1
            dscores = dscores / num_examples
            dW, db = self.backprop(df, dscores, zvec)
            # Update! TODO: Indexing....
            for n in range(1, self.num_layers):
                self.layers[n-1].update(dW[n-1], db[n-1], self.step_size)

        # Calculate the final weights and return
        W = []
        b = []
        for n in range(self.num_layers-1):
            W.append(self.layers[n].W)
            b.append(self.layers[n].b)

        return (W, b)


def layer_size_test(X, y, D):
    # X - data
    # y - labels

    layer_list = []
    bias_list = []
    zlist = []
    num_layers = 3
    layer_dims = np.array([100, 200, 50])

    # Create layers
    for n in range(num_layers):
        if(n == 0):
            W = np.random.randn(D, layer_dims[n])
        else:
            W = np.random.randn(layer_dims[n-1], layer_dims[n])
        b = np.zeros((1,layer_dims[n]))
        # create an output layer at the end
        #if(n == num_layers-1):
        #    W = np.zeros((layer_dims[n-1], layer_dims[n]))
        layer_list.append(W)
        bias_list.append(b)
        # calculate Z size
        z = np.zeros((X.shape[0], layer_dims[n]))
        zlist.append(z)

    # Forward pass
    for n in range(num_layers-1):
        if(n == 0):
            activ = X
        else:
            activ = zlist[n-1]
        zlist[n] = np.dot(activ, layer_list[n]) + bias_list[n]

    print("done")



# ======== ENTRY POINT ======== #
if __name__ == "__main__":

    N = 500
    h = 100
    D = 2
    K = 3
    theta = 0.3
    spiral_data = nnu.create_spiral_data(N, D, K, theta)

    ref_net = rc.TwoLayerNetSingleFunction()
    num_layers = 2
    layer_dims = []
    layer_sizes = []
    for n in range(num_layers-1):
        layer_dims.append(D)
        layer_sizes.append(h)



    input_dim = D
    #mod_net = ModularNetwork(layer_dims, layer_sizes, input_dim)
    mod_net = LessModularNetwork(layer_dims, layer_sizes)
    mod_net.verbose = True
    #net.init_params(h, D, K)
    num_iters = 300

    X = spiral_data[0]      # data
    y = spiral_data[1]      # labels

    # TODO ; Make sure that the size of the layers is calculated correctly
    layer_size_test(X, y, D)

    #NNFunction(X, y)
    W1, W2, b1, b2 = ref_net.train(X, y, D, h, K, num_iters)
    modW, modb = mod_net.train(X, y, num_iters)

    # Visualize the classifier
    h = 0.02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    #W = (net.W1, net.W2)
    #b = (net.b1, net.b2)
    W = (modW[0], modW[1])
    b = (modb[0], modb[1])
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

    # To save time copy and paste but with a new W
#    W = (modW[0], modW[1])
#    b = (modb[0], modb[1])

#    if(type(W) is tuple):
#        Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W[0]) + b[0]), W[1]) + b[1]
#    else:
#        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
#    Z = np.argmax(Z, axis=1)
#    Z = Z.reshape(xx.shape)
#
#    #self.fig_classifier = plt.figure()
#    plt.figure(1)
#    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
#    plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.show()
#
