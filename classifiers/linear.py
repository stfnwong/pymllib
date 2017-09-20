"""
LINEAR CLASSIFIER

"""

import numpy as np


class LinearClassifier(object):
    def __init__(self):
        self.step_size = 1e-0
        self.reg = 1e-3         # regularization strength
        self.num_iter = 200
        self.W = None
        self.scores = None
        self.dscores = None
        self.loss = None

    def init_params(self, D, K):

        # K = number of classes
        # D = dimension of data

        self.W = 0.01 * np.random.randn(D, K)
        self.b = np.zeros((1,K))

    def forward_iter(self, X, y):
        num_examples = X.shape[0]
        scores = np.dot(X, self.W) + self.b
        self.scores = scores

        # Compute class probs
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * self.reg * np.sum(self.W * self.W)
        loss = data_loss + reg_loss

        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # Store internal data
        self.loss = loss
        self.dscores = dscores

        return loss, dscores

    def backward_iter(self, X):
        # backprop the gradient params
        dW = np.dot(X.T, self.scores)
        db = np.sum(self.scores, axis=0, keepdims=True)

        dW += self.reg * self.W

        Wout = self.W - self.step_size * dW
        bout = self.b - self.step_size * db

        return Wout, bout

    """ ================ DEPRECATED ================ """
    # Just do inner loop here, outer loop can happen in GUI
    # TODO ; Split into forward_iter() and backward_iter() ?
    def compute_loss_iter(self, W, X, y):

        num_examples = X.shape[0]
        self.scores = np.dot(X, W) + self.b

        # Compute class probs
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * self.reg * np.sum(W * W)
        loss = data_loss + reg_loss

        # Compute gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backprop the gradient params
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        dW += self.reg * W

        Wout = W - self.step_size * dW
        bout = self.b - self.step_size * db

        return loss, Wout, bout
