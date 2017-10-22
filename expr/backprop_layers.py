"""
Backprop with layer objects

Stefan Wong 2017
"""

import os
import sys
# TODO : Need to get rid of these
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
import numpy as np
import neural_net_utils as nnu

# Use the layers in layers folder
from layers import AffineLayer, ReLULayer, SigmoidLayer
# Debug
from pudb import set_trace; set_trace()

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
        self.layers = self.init_layers(internal_layer_dims, layer_types)
        self.num_layers = len(self.layers)
        #self.num_layers = len(layer_dims) + 1

        # Pre-allocate memory for forward and backward activations
        # TODO : get rid of these, cache the forward activation in the layers
        #self.z_forward = []
        #self.z_backward = []    # Gradient on weights
        #self.b_backward = []    # Gradient on biases
        #for l in range(len(self.layers)):
        #    z = np.zeros((input_dim, internal_layer_dims[l+1]))
        #    b = np.zeros((1, internal_layer_dims[l+1]))
        #    dz = np.zeros((internal_layer_dims[l], internal_layer_dims[l+1]))
        #    self.z_forward.append(z)
        #    self.z_backward.append(dz)
        #    self.b_backward.append(b)

        # Debug options
        self.verbose = False
        self.cache_loss = False

    def __str__(self):
        s = []
        s.append('%d layers network\n' % (self.num_layers))
        n = 1
        for l in self.layers:
            s.append('%2d) %s' % (n, str(l)))
            n += 1
        s.append('\n')

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

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
        layers = []
        for l in range(len(layer_types)):
            dim_idx = l+1
            # TODO: move sizes out to local variables
            if(layer_types[l] == 'relu'):
                layer = ReLULayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            elif(layer_types[l] == 'sigmoid'):
                layer = SigmoidLayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            elif(layer_types[l] == 'linear'):
                layer = AffineLayer(layer_dims[dim_idx-1], layer_dims[dim_idx])
            else:
                print('Invalid layer type %s' % (layer_types[l]))
                sys.exit(2)     # TODO : throw an exception?
            layers.append(layer)

        return layers

    def gradient_descent(self, X, y, cache_a=False):

        # For debugging - remove in final version
        if(cache_a):
            a_cache = []

        N = X.shape[0]
        # ==== Forward pass
        activation = X
        for l in range(len(self.layers)):
            activation = self.layers[l].forward(activation)
            if(cache_a is True):
                a_cache.append(activation)
            #self.z_forward[l] = z_temp
            #z_input = self.z_forward[l]

        # Compute scores
        exp_scores = np.exp(self.layers[-1].Z)
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
        #self.z_backward[-1] = dscores

        # ==== Backward pass
        dz_input = dscores
        for l in range(len(self.layers)-1):
            idx = len(self.layers) - l - 1          # should be N, .., 2
            #dz_temp = self.layers[idx].backward(dz_input, self.z_forward[idx])
            if(idx == 2):
                prev_activ = X
            else:
                prev_activ = self.layers[idx-1].Z

            dx, dw, db = self.layers[idx].backward(dz_input, prev_activ) # TODO : Need the activation from the layer before?
            self.layers[idx].update(dw, db, self.step_size)
            #self.z_backward[idx] = np.dot(self.layers[idx].W.T, self.z_backward[idx+1]) * dz_temp
            #self.z_backward[idx] = dz_temp
            #db_temp = np.sum(dz_input)
            #self.b_backward[idx] = db_temp
            #dz_input = self.z_backward[idx]

        # Add regularization gradient contribution
        for l in range(len(self.layers)):
            self.z_backward[l] += self.reg * self.layers[l].W

        if(cache_a is True):
            return loss, a_cache
        else:
            return loss

    def train(self, X, y, num_iter=10000, debug=False):

        if(self.cache_loss):
            loss_cache = np.ndarray((1, num_iter))

        for i in range(num_iter):
            if(debug is True):
                loss, a_cache = self.gradient_descent(X, y, debug)
            else:
                loss = self.gradient_descent(X, y, debug)
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


# ==== TEST CODE ==== #
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

    loss_cache = net.train(X, y, num_iter=num_iter, debug=True)

    # TODO : plot the loss funtion
    #import matplotlib.pyplot as plt
    print(net)
    print(len(loss_cache))


if __name__ == "__main__":
    main()
