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

# TODO ; Move to classifier folder?
"""
FCNET : This is a neural net using an architecture more similar to that in
CS231ncc
"""
class FCNet(object):
    def __init__(self, hidden_dims, input_dim, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float332, seed=None,
                 verbose=False):

        self.verbose = verbose
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Initialize the parameters of the network, storing all values into a
        # dictionary at self.params. The keys to the dictionary are W1, b1 fo
        # layer 1, W2, b2 for layer 2, and so on.
        if type(hidden_dims) is not list:
            raise ValueError('hidden_dim must be a list')

        self.N = input_dim
        self.C = num_clases
        dims = [self.N] + hidden_dims + [self.C]
        # Weight dict
        Ws = {'W' + str(i+1) : weight_scale * np.random.randn(dims[i], dims[i+1]) for i in range(len(dims)-1)}
        bs = {'b' + str(i+1) : np.zeros(dims[i+1]) for i in range(len(dims)-1)}

        self.params.update(bs)
        self.params.update(Ws)

        # When using dropout, we must pass a dropout_param dict to each dropout
        # layer so that the layer knows the dropout probability and the mode
        # (train/test).
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode' : 'train', 'p' : dropout}
            if self.verbose:
                print("Using dropout with p = %f" % (self.dropout_param['p']))
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. We pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the
        # forward pass of the second batch norm layer, and so on
        if self.use_batchnorm:
            self.bn_params = {'bn_params' + str(i+1) : {'mode' : 'train',
                                                        'running_mean' : np.zeros(dims[i+1]),
                                                        'running_var'  : np.zeros(dims[i+1])}
                              for i in range(len(dims)-2) }
            gammas = {'gamma' + str(i+1) : np.ones(dims[i+1])
                      for i in range(len(dims)-2)}
            betas = {'beta' + str(i+1) : np.zeros(dims[i+1])
                     for i in range(len(dims)-2)}
            self.params.update(gammas)
            self.params.update(betas)

        # Cast params to correct data type
        if(self.verbose):
            print("Casting parameters to type %s" % (self.dtype))
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(self.dtype)

    def loss(self, X, y=None):
        """
        LOSS

        Compute loss and gradients for the fully connected network
        """

        X = X.astype(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        # Set batchnorm params based on whether this is a training or a test
        # run
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for k, bn_param in self.bn_params.iteritems():
                bn_param[mode] = mode

        # ===============================
        # FORWARD PASS
        # ===============================
        hidden = {}
        hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))   # TODO ; Check this...

        if self.use_dropout:

            hdrop, cache_hdrop = dropout_forward(hidden['h0'],
                                                 self.dropout_param)
            hidden['hdrop0'] = hdrop
            hidden['cache_hdrop0'] = cache_hdrop

        # Iterate over layers
        for l in range(self.num_layers):
            idx = l + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]

            if self.use_dropout:
                h = hidden['hdrop' + str(idx-1)]
            else:
                h = hidden['h' + str(idx-1)]

            if self.use_batchnorm:
                gamma = self.params['gamma' + str(idx)]
                beta = self.params['beta' + str(idx)]
                bn_param = self.bn_params['bn_paramm' + str(idx)]

            # Compute the forward pass
            # output layer is a special case
            if idx == self.num_layers:
                # TODO ; re-create the layer definitions
                h, cache_h = flayers.affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache_h' + str(idx)] = cache_h
            else:
                if self.use_batchnorm:
                    h, cache_h = flayers.affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
                    hidden['h' + str(idx)] = h
                    hidden['cache_h' + str(idx)] = cache_h














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
