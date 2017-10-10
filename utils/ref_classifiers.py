"""
TWO LAYER REF
The "reference" implementation from CS231n, as well as the
"reference" linear classifier. The iteration should be done
outside the loop so that we can 'animate' the convergence
of the network over time.

Stefan Wong 2017
"""

import numpy as np

# Debug
from pudb import set_trace; set_trace()

class TwoLayerNetSingleFunction(object):
    def __init__(self, reg=1e-3, step_size=1e0):
        self.reg = reg
        self.step_size = step_size
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None

    def train(self, X, y, D, h, K, num_iter=10000):
            # initialize parameters randomly
        h = 100 # size of hidden layer
        W1 = 0.01 * np.random.randn(D,h)
        b1 = np.zeros((1,h))
        W2 = 0.01 * np.random.randn(h,K)
        b2 = np.zeros((1,K))

        # some hyperparameters
        step_size = 1e-0
        reg = 1e-5 # regularization strength
        # gradient descent loop
        num_examples = X.shape[0]
        for i in range(num_iter):

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
            if i % 100 == 0:
                print("iteration %d: loss %f" % (i, loss))

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

        # Store the final weight values
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

        return W1, W2, b1, b2

"""
Basic two layer network as a single function
"""
def RefTwoLayerNetwork(params, X, y, reg=1e-3, step_size=1e-1, h=100, D=2, K=2):

    num_examples = X.shape[0]
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

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
    # finally into W,b
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # perform a parameter update
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

    out_params = {}
    out_params['loss'] = loss
    out_params['W1'] = W1
    out_params['W2'] = W2
    out_params['b1'] = b1
    out_params['b2'] = b2

    return out_params


def LinearReference(W, X, y, reg=1e-3, step_size=1e-1):
    num_examples = X.shape[0]

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg*W # regularization gradient

    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

    return loss, W, b

