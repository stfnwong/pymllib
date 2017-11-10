"""
NEURAL NET CLASSIFIER

Stefan Wong 2017
"""

import numpy as np

# debug
from pudb import set_trace; set_trace()

# Some simple activation functions here
def relu(z):
    return np.maximum(0, z)

def relu_dz(dz, X):
    d = np.zeros_like(dz)
    d[X > 0] = 1
    return d * dz

class NeuralNetwork(object):
    def __init__(self, hidden_dims, input_dim, num_iter=10000, reg=1e-0, step_size=1e-3):
        # Hyper parameters
        self.reg = reg
        self.step_size = step_size
        self.num_iter = num_iter
        # Layers
        self.W = []         # Layer weights
        self.Z = []         # Activations
        self.b = []         # Biases
        self.num_layers = len(hidden_dims) + 1
        # Debug params
        self.verbose = False
        self.cache_loss = False

        # Setup layers
        self.init_layers(hidden_dims, input_dim)

    def __str__(self):
        s = []
        s.append("%d layer neural network (%d hidden layers)\n" % (self.num_layers, len(self.W)))
        n = 0
        for l in self.W:
            s.append("\tLayer %d : size %d\n" % (n+1, self.W[n].shape[1]))
            n += 1

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def init_layers(self, hidden_dims, input_dim):

        for n in range(len(hidden_dims)):
            if n == 0:
                W = 0.01 * np.random.randn(input_dim, hidden_dims[n])
            else:
                W = 0.01 * np.random.randn(hidden_dims[n-1], hidden_dims[n])
            #W = 0.01 * np.random.randn(input_dim, hidden_dims[n-1])
            b = np.zeros((1, hidden_dims[n]))
            self.W.append(W)
            self.b.append(b)

        # activation before non-linearity
        for n in range(self.num_layers-1):
            z = np.zeros((input_dim, self.W[n].shape[1]))
            self.Z.append(z)

    def backprop(self, X, y, f, df):

        num_examples = X.shape[0]
        # TODO : Move this into self.dW, self.db
        dW = [np.zeros(w.shape) for w in self.W]
        db = [np.zeros(b.shape) for b in self.b]

        loss = 0
        self.Z[0] = X
        scores = []
        # Forward pass
        for n in range(len(self.W)):
            if(n == 0):
                z = np.dot(X, self.W[n]) + self.b[n]
            else:
                z = np.dot(self.Z[n-1], self.W[n]) + self.b[n]
            self.Z[n] = f(z)
            scores.append(self.Z[n])  # TODO : redundant...
            #activation = self.Z[n]

        #z[0] = np.dot(X, W[0]) + b[0]
        #z[1] = np.dot(z[0], W[1]) + b[1]
        #z[2] = np.dot(z[1], W[2]) + b[2]
        ##... etc

        # Last activation is the final score
        probs = np.exp(scores[-1]) / np.sum(np.exp(scores[-1]), axis=1, keepdims=True)
        logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(logprobs) / num_examples
        reg_loss = 0
        for n in range(len(self.W)-1):
            reg_loss += 0.5 * self.reg * np.sum(self.W[n] * self.W[n])
        loss = data_loss + reg_loss

        # Backward pass
        #delta = (self.Z[-1] - y) * df(self.Z[-1])
        delta = probs
        delta[range(num_examples), y] -= 1
        delta /= num_examples
        for n in range(self.num_layers-1, 2, -1):
            if(self.verbose):
                print("Backward pass in layer %d of %d" % (n, self.num_layers-1))

            dz = df(self.Z[n])
            delta = np.dot(delta, self.W[n].T) * dz
            db[n] = delta * self.reg
            dW[n] = np.dot(scores[n], self.W[n].T) * self.reg

        return dW, db, loss

    def train(self, X, y):

        if(self.cache_loss):
            loss_cache = np.zeros((1, self.num_iter))

        for n in range(self.num_iter):

            dW, db, loss = self.backprop(X, y, relu, relu_dz)
            self.update(dW, db)
            if(self.cache_loss):
                loss_cache[n] = loss
            if(self.verbose):
                if(n % 100 == 0):
                    print("iter %d : loss = %f" % (n, loss))

        if(self.cache_loss):
            return loss_cache


# For now this is a 2-layer network, TODO : Generalize
class OldNeuralNetwork(object):
    def __init__(self, h=100, reg=0.05, ss=1e-3):
        self.h_layer_size = h
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.hidden_layer = None
        # hyperparams
        self.reg = reg
        self.step_size = ss
        self.ss_decay = 0.96
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

        self.W1 += -self.step_size * dW1
        self.b1 += -self.step_size * db1
        self.W2 += -self.step_size * dW2
        self.b2 += -self.step_size * db2
        #self.step_size *= self.ss_decay

        return (dW1, dW2), (db1, db2)

    # TODO ; Do the validation outside this 'loop'
    def train_iter(self, X, y, loss, grads):
        # perform
        self.W1 -= self.step_size * grads['dW'][0]
        self.b1 -= self.step_size * grads['db'][0]
        self.W2 -= self.step_size * grads['dW'][1]
        self.b2 -= self.step_size * grads['db'][1]
        self.step_size += self.ss_decay

    def predict_iter(self, X):

        z = np.dot(X, self.W1) + self.b1
        h = np.maximum(z, 0)
        out = np.dot(h, self.W2) + self.b2
        y_pred = np.argmax(out, axis=1)

        return y_pred


if __name__ == "__main__":


    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
    import neural_net_utils as nnu

    N = 500
    h = 100
    D = 2
    K = 3
    theta = 0.3
    spiral_data = nnu.create_spiral_data(N, D, K, theta)
    # Get some data
    X = spiral_data[0]
    y = spiral_data[1]

    # Start with one hidden layer
    layer_dims = [h, K]
    input_dim = D
    nn = NeuralNetwork(layer_dims, input_dim)
    print(nn)
    nn.verbose = True
    nn.train(X, y)

    print(nn)
