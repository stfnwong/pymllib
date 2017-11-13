"""
CONV_PARAM_SEARCH
<<<<<<< HEAD
Find suitable hyperparameters for a convnet

=======

Find suitable parameters for convolutional neural network
>>>>>>> 544613aad776e6d65123884fa5e527a85e852e23
Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# Lib
from pymllib.solver import solver
from pymllib.classifiers import convnet
from pymllib.utils import data_utils

# Debug
from pudb import set_trace; set_trace()

class ConvParamSearch(object):
    def __init__(self, **kwargs):
        # Reserve names for model and solver
        self.solv = None
        self.model = None
        self.dataset = None
        self.train_data = None
        # Search params
        self.max_searches = kwargs.pop('max_searches', 1e6)
        self.num_train = kwargs.pop('num_train', 500)
        self.ws_range = kwargs.pop('ws_range', [-6, 1])
        self.lr_range = kwargs.pop('lr_range', [-5, 1])
        self.reg_range = kwargs.pop('reg_range', [-3, 1])
        # Model params
        self.model_input_dim = kwargs.pop('input_dim', (3, 32, 32))
        self.model_hidden_dims = kwargs.pop('hidden_dims', [256, 100])
        self.model_num_filters = kwargs.pop('num_filters', [16, 32])
        self.model_num_classes = kwargs.pop('num_classes', 10)
        # Solver params
        self.solver_num_epochs = kwargs.pop('num_epochs', 100)
        self.solver_batch_size = kwargs.pop('batch_size', 20)
        self.solver_update_rule = kwargs.pop('update_rule', 'adam')
        self.solver_checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.solver_checkpoint_dir = kwargs.pop('checkpoint_dir', '.')
        self.solver_print_every = kwargs.pop('print_every', 20)
        # Output params
        self.lr_output = 0.0
        self.ws_output = 0.0
        self.reg_output = 0.0
        # Other params
        self.verbose = kwargs.pop('verbose', False)

    def __str__(self):
        s = []
        s.append("Model parameters")

        if self.model is None:
            # Make a model to simplify printing
            model = convnet.ConvNetLayer(input_dim=self.model_input_dim,
                                         hidden_dims=self.model_hidden_dims,
                                         num_filters=self.model_num_filters)
        else:
            model = self.model
        s.append(str(model))
        s.append("Solver parameters")

        if self.solv is None:
            solv = solver.Solver(model,
                                 None,
                                 num_epochs=self.solver_num_epochs,
                                 batch_size=self.solver_batch_size,
                                 update_rule=self.solver_update_rule,
                                 #optim_config={'learning_rate': learning_rate},
                                 verbose=self.verbose,
                                 print_every=self.solver_print_every,
                                 checkpoint_name=self.solver_checkpoint_name,
                                 checkpoint_dir=self.solver_checkpoint_dir)
        else:
            solv = self.solv
        s.append(str(solv))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def load_data(self, data_dir):
        self.dataset = data_utils.get_CIFAR10_data(data_dir)
        self.train_data = {
            'X_train': self.dataset['X_train'][:self.num_train],
            'y_train': self.dataset['y_train'][:self.num_train],
            'X_val':   self.dataset['X_val'][:self.num_train],
            'y_val':   self.dataset['y_val'][:self.num_train]
        }

        if self.verbose:
            for k, v in self.dataset.items():
                print("%s : %s " % (k, v.shape))

    def param_search(self):
        """
        PARAM_SEARCH

        Attempt to find parameters that allow the network to train
        """

        if self.verbose:
            print("Searching for learning rates in range  %f - %f" % (10**(self.lr_range[0]), 10**(self.lr_range[1])))
            print("Searching for weight scales in range   %f - %f" % (10**(self.ws_range[0]), 10**(self.ws_range[1])))
            print("Searching for regularization in range  %f - %f" % (10**(self.reg_range[0]), 10**(self.reg_range[1])))

        param_search = True
        n = 0
        while param_search and n < self.max_searches:
            weight_scale = 10 ** (np.random.uniform(self.ws_range[0], self.ws_range[1]))
            learning_rate = 10 ** (np.random.uniform(self.lr_range[0], self.lr_range[1]))
            reg = 10**(np.random.uniform(self.reg_range[0], self.reg_range[1]))

            self.model = convnet.ConvNetLayer(input_dim=self.model_input_dim,
                                              hidden_dims=self.model_hidden_dims,
                                              num_filters=self.model_num_filters,
                                              reg=reg,
                                              weight_scale=weight_scale,
                                              verbose=self.verbose)
            if self.verbose:
                print(self.model)
            self.solv = solver.Solver(self.model,
                                      self.train_data,
                                      num_epochs=self.solver_num_epochs,
                                      batch_size=self.solver_batch_size,
                                      update_rule=self.solver_update_rule,
                                      optim_config={'learning_rate': learning_rate},
                                      verbose=self.verbose,
                                      print_every=self.solver_print_every,
                                      checkpoint_name=self.solver_checkpoint_name,
                                      checkpoint_dir=self.solver_checkpoint_dir)
            self.solv.train()
            n += 1
            if max(self.model.train_acc_history) >= 1.0:
                param_search = False
                self.lr_output = learning_rate
                self.ws_output = weight_scale
                self.reg_output = reg
                print("Found parameters after %d epochs total (%d searches of %d epochs each)" % (n * self.solver_num_epochs, n, self.solver_num_epochs))
                return

        print("Failed to train to accuracy of 1.0 in %d searches of %d epochs" % (n, self.solver_num_epochs))
        print("Last learning rate : %f" % learning_rate)
        print("Last weight scale  : %f" % weight_scale)
        print("Last reg           : %f" % reg)


# Basic test
if __name__ == "__main__":
    data_dir = 'datasets/cifar-10-batches-py'
    searcher = ConvParamSearch(lr_range=[-6, -1],
                               ws_range=[-5, -1],
                               reg_range=[-3, -1],
                               num_train=800,
                               verbose=True)
    print(searcher)
    searcher.load_data(data_dir)
    searcher.param_search()
