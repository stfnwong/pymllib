"""
Train a 2-layer network on spiral data

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import numpy as np
import matplotlib.pyplot as plt
# Internal libs
import neural_net_utils as nnu
import activations as activ
#Debug
from pudb import set_trace; set_trace()

class NeuralNet(object):
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.grads = None

        # This is just a two layer network
        self.scores = None
        self.loss = 0
        # Hyperparams
        self.reg = 0
        self.step_size = 0
        self.num_iter = 5000

    def init_params(self, h, D, K, reg=1e-3, step_size=1e-0):
        # D - dimension of data
        # K - number of classes in dataset

        self.reg = reg
        self.step_size = step_size
        # init weights
        self.W1 = 0.001 * np.random.randn(D, h)
        self.b1 = np.zeros((1,h))
        self.W2 = 0.001 * np.random.randn(h,K)
        self.b2 = np.zeros((1,K))
        # Set gradients to zero
        self.grads = {}
        self.grads['dW1'] = 0
        self.grads['dW2'] = 0
        self.grads['db1'] = 0
        self.grads['db2'] = 0

    # TODO : Something strange here about the class scores
    def grad_descent(self, X, y):

        num_examples = X.shape[0]

        print('reg ; %f' % self.reg)
        print('ss  : %f' % self.step_size)

        for n in range(self.num_iter):

            hidden_layer = activ.relu(np.dot(X, self.W1) + self.b1)
            #hidden_layer = np.maximum(0, np.dot(X, self.W1) + self.b1) # threshold = ReLU
            scores = np.dot(hidden_layer, self.W2) + self.b2
            # Compute class probs
            exp_scores = np.exp(scores)
            # For display in debugger watch window
            max_score = np.max(scores)
            max_exp = np.max(exp_scores)
            #return
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            max_probs = np.max(probs)

            # Compute the loss
            correct_logprobs = -np.log(probs[range(num_examples), y])
            data_loss = np.sum(correct_logprobs) / num_examples
            reg_loss = 0.5 * self.reg * np.sum(self.W1 * self.W1) + 0.5 * self.reg * np.sum(self.W2 * self.W2)
            self.loss = data_loss + reg_loss

            # Training debug - remove
            if(n % 10 == 0):
                print('iter %d : loss = %f' % (n, self.loss))
                print('grads : %s' % self.grads)

            dscores = probs
            dscores[range(num_examples), y] = -1
            dscores /= num_examples

            # Backprop the gradient to the params
            self.grads['dW2'] = np.dot(hidden_layer.T, dscores)
            self.grads['db2'] = np.sum(dscores, axis=0, keepdims=True)
            dhidden = np.dot(dscores, self.W2.T)

            # backprop ReLU non-linearity
            dhidden = dhidden * activ.relu_dz(hidden_layer)
            #dhidden[hidden_layer <= 0] = 0
            self.grads['dW1'] = np.dot(X.T, dhidden)
            self.grads['db1'] = np.sum(dhidden, axis=0, keepdims=True)

            # Add reg gradient contribution
            self.grads['dW2'] += self.reg * self.W2
            self.grads['dW1'] += self.reg * self.W1

            # Update params
            self.W1 += -self.step_size * self.grads['dW1']
            self.b1 += -self.step_size * self.grads['db1']
            self.W2 += -self.step_size * self.grads['dW2']
            self.b2 += -self.step_size * self.grads['db2']

        return self.loss, self.grads


class TwoLayerNet(object):

    def __init__(self, reg=1e-3, step_size=1e0):
        self.reg = reg
        self.step_size = step_size

    def train(self, X, y):
            # initialize parameters randomly
        h = 100 # size of hidden layer
        W1 = 0.01 * np.random.randn(D,h)
        b1 = np.zeros((1,h))
        W2 = 0.01 * np.random.randn(h,K)
        b2 = np.zeros((1,K))

        # some hyperparameters
        step_size = 1e-0
        reg = 1e-3 # regularization strength

        # gradient descent loop
        num_examples = X.shape[0]
        for i in range(10000):

            # evaluate class scores, [N x K]
            hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
            scores = np.dot(hidden_layer, W2) + b2

            # compute the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

            # compute the loss: average cross-entropy loss and regularization
            corect_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(corect_logprobs)/num_examples
            reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
            loss = data_loss + reg_loss
            if i % 1000 == 0:
                print("iteration %d: loss %f" % (i, loss))

            # max scores
            max_scores = np.max(scores)
            max_exp = np.max(exp_scores)

            # compute the gradient on scores
            dscores = probs
            dscores[range(num_examples),y] -= 1
            dscores /= num_examples

            # backpropate the gradient to the parameters
            # first backprop into parameters W2 and b2
            dW2 = np.dot(hidden_layer.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            # next backprop into hidden layer
            dhidden = np.dot(dscores, W2.T)
            # backprop the ReLU non-linearity
            dhidden[hidden_layer <= 0] = 0
            # finally into W1,b1
            dW = np.dot(X.T, dhidden)
            db = np.sum(dhidden, axis=0, keepdims=True)

            # add regularization gradient contribution
            dW2 += reg * W2
            dW += reg * W1

            # perform a parameter update
            W1 += -step_size * dW
            b1 += -step_size * db
            W2 += -step_size * dW2
            b2 += -step_size * db2

        return W1, W2, b1, b2



# This is just a copy paste of the version from CS231n
def NNFunction(X,y):
        # initialize parameters randomly
    h = 100 # size of hidden layer
    W1 = 0.01 * np.random.randn(D,h)
    b1 = np.zeros((1,h))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3 # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(10000):

        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1) # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples),y])
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # max scores
        max_scores = np.max(scores)
        max_exp = np.max(exp_scores)

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),y] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W1,b1
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W1

        # perform a parameter update
        W1 += -step_size * dW
        b1 += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    # This is the h for the visusalization, not for the size of the hidden
    # layer
    h = 0.02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

    W =(W1, W2)
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



# ======== ENTRY POINT ======== #
if __name__ == "__main__":

    # TODO : need data
    N = 500
    h = 100
    D = 2
    K = 3
    theta = 0.3
    spiral_data = nnu.create_spiral_data(N, D, K, theta)

    net = TwoLayerNet()
    #net.init_params(h, D, K)
    num_iters = 10000

    X = spiral_data[0]
    y = spiral_data[1]

    #NNFunction(X, y)
    W1, W2, b1, b2 = net.train(X, y)


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
