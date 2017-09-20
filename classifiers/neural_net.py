"""
NEURAL NET CLASSIFIER


Stefan Wong 2017
"""

import numpy as np

# For now this is a 2-layer network, TODO : Generalize
class NeuralNetwork(object):
    def __init__(self, h=100, reg=1e-3, ss=1e-0):
        self.h_layer_size = h
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.hidden_layer = None
        # hyperparams
        self.reg = reg
        self.step_size = ss
        # layer outputs
        self.dscores = None
        self.loss = 0


    def init_params(self, D, K):
        # D - dimension of data
        # K - number of classes

        h = self.h_layer_size
        self.W1 = 0.01 * np.random.randn(D, h)
        self.W2 = 0.01 * np.random.randn(h, K)
        self.b1 = np.zeros((1,h))
        self.b2 = np.zeros((1,K))

    def forward_iter(self, X, y):

        num_examples = X.shape[0]

        # ReLU activation
        self.hidden_layer = np.maximum(0, np.dot(X, self.W1) + self.b1)
        scores = np.dot(self.hidden_layer, self.W2) + self.b2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        # compute the loss, avg cross entropy loss and reg
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * self.reg * np.sum(self.W1 * self.W1) + 0.5 * self.reg * np.sum(self.W2 * self.W2)
        loss = data_loss + reg_loss

        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        self.loss = loss
        self.dscores = dscores

        return loss, dscores


    def backward_iter(self, X):

        dW2 = np.dot(self.hidden_layer.T, self.dscores)
        db2 = np.sum(self.dscores, axis=0, keepdims=True)

        dhidden = np.dot(self.dscores, self.W2.T)
        dhidden[self.hidden_layer <= 0] = 0
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        # Add reg gradient
        dW2 += self.reg * self.W2
        dW1 += self.reg * self.W1

        # perform
        self.W1 -= self.step_size * dW1
        self.b1 -= self.step_size * db1
        self.W2 -= self.step_size * dW2
        self.b2 -= self.step_size * db2

        return (dW1, dW2), (db1, db2)




