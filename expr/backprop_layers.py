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
        s.append('Linear Layer, \n\tinput dim : %d, \n\tlayer size : %d\n' % (self.W.shape[0], self.W.shape[1]))
        return ''.join(s)

    def __repr__(self):
        return self.__str__()

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
        return self.__str__()

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
        return self.__str__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, dz, X_prev):  # Here to keep function prototype symmetry
        return dz * (1 - dz)


# ========= NETWORK ======== #
"""
Basic layered Neural Network using new layer objects
"""
class LayeredNeuralNetwork(object):
    def __init__(self, input_dim, layer_dims, layer_types=None, reg=1e-0, step_size=1e-3):
        """
        TODO : Full doc string

        input_dim:
            Size of input

        layer_dims:
            List of sizes for each hidden layer. layer_dims[-1] is the size of the output
            layer.
        """
        # Hyperparameters
        self.reg = reg
        self.step_size = step_size
        # Init layers
        if(layer_types is None):
            layer_types = ['linear'] * len(layer_dims)
        else:
            assert len(layer_types) == len(layer_dims)

        #internal_layer_dims = [input_dim].extend(layer_dims)
        internal_layer_dims = []
        internal_layer_dims.append(input_dim)
        for t in layer_dims:
            internal_layer_dims.append(t)
        self.init_layers(internal_layer_dims, layer_types)
        self.num_layers = len(self.layers)
        #self.num_layers = len(layer_dims) + 1

        # Pre-allocate memory for forward and backward activations
        self.z_forward = []
        self.z_backward = []    # Gradient on weights
        self.b_backward = []    # Gradient on biases
        for l in range(len(self.layers)):
            z = np.zeros((input_dim, internal_layer_dims[l+1]))
            b = np.zeros((1, internal_layer_dims[l+1]))
            dz = np.zeros((internal_layer_dims[l], internal_layer_dims[l+1]))
            self.z_forward.append(z)
            self.z_backward.append(dz)
            self.b_backward.append(b)

        # Debug options
        self.verbose = False
        self.cache_loss = False

    def __str__(self):
        s = []
        s.append('% layers\n' % (self.num_layers))
        for l in self.layers:
            s.append('%s' % (str(l)))
        s.append('\n')

        return ''.join(s)

    def __repr__(self):
        return self.__str__

    def init_layers(self, layer_dims, layer_types):
        """
        INPUTS:
            layer_dims:
                list containing dimensions for each layer. layer_dims[0]
                must be an integer representing the size of the input
                presented to the first layer. layer_dims[-1] must be an
                integer representing the size of the output on the last
                layer

            layer_types:
                A list indicating the type of activation for each layer.
                This length of this list must be equal to the length of
                layer_dims.

        """
        self.layers = []
        # TODO : if I use soemthing like
        # for l in self.layers;
        # del(l)
        # does this improve memory usage if/when we reset the weights?

        for l in range(len(layer_types)):
            dim_idx = l+1
            # TODO: move sizes out to local variables
            if(layer_types[l] == 'relu'):
                layer = ReLULayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            elif(layer_types[l] == 'sigmoid'):
                layer = SigmoidLayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            elif(layer_types[l] == 'linear'):
                layer = LinearLayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            else:
                print('Invalid layer type %s' % (layer_types[l]))
                sys.exit(2)     # TODO : throw an exception?

            self.layers.append(layer)

    def gradient_descent(self, X, y):

        N = X.shape[0]
        # ==== Forward passc
        z_input = X
        for l in range(len(self.layers)):
            z_temp = self.layers[l].forward(z_input)
            self.z_forward[l] = z_temp
            z_input = self.z_forward[l]

        # Compute scores
        exp_scores = np.exp(self.z_forward[-1])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        logprobs = -np.log(probs[range(N), y])
        # Compute loss
        data_loss = np.sum(logprobs) / N
        reg_loss = 0
        for l in range(len(self.layers)):
            reg_loss += 0.5 * self.reg * np.sum(self.layers[l].W * self.layers[l].W)
        loss = data_loss + reg_loss

        # Compute derivative of scores
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        # ==== Backward pass
        dz_input = dscores
        for l in range(len(self.layers)):
            idx = len(self.layers) - l - 1
            # TODO : I am not correctly applying the backward pass.... I think
            dz_temp = self.layers[idx].backward(dz_input, self.z_forward[idx])
            self.z_backward[idx] = dz_temp
            db_temp = np.sum(dz_input)
            self.b_backward[idx] = db_temp
            dz_input = self.z_backward[idx]

        # Add regularization gradient contribution
        for l in range(len(self.layers)):
            self.z_backward[l] += self.reg * self.layers[l].W

        return loss

    def train(self, X, y, num_iter=10000):

        if(self.cache_loss):
            loss_cache = np.ndarray((1, num_iter))

        for i in range(num_iter):
            loss = self.gradient_descent(X, y)
            if(self.cache_loss):
                loss_cache[i] = loss
            if(self.verbose):
                if(i % 100 == 0):
                    print('iter %d : loss %f' % (i, loss))
            self.update_weights()

        if(self.cache_loss):
            return loss_cache
        else:
            return None


    def update_weights(self):
        for l in range(len(self.layers)):
            self.layers[l].update(self.z_backward[l], self.b_backward[l], self.step_size)



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

    # Hyperparams
    reg = 1e-0
    step_size = 1e-3
    num_iter = 10000

    # Create layers
    layer_sizes = [h, 3]
    layer_types = ['relu', 'linear']

    net = LayeredNeuralNetwork(D, layer_sizes, layer_types, reg=reg, step_size=step_size)
    net.verbose = True
    net.cache_loss = True
    print(net)
    loss_cache = net.train(X, y, num_iter=num_iter)

    print(net)
    print(len(loss_cache))





if __name__ == "__main__":
    main()
