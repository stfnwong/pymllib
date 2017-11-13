"""
CONV_PARAM_SEARCH

Find suitable parameters for convolutional neural network
Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
# Lib
from pymllib.solver import solver
from pymllib.classifiers import convnet
from pymllib.util import data_utils


class ConvParamSearch(object):

    def __init__(self, max_searches=1e6):
        self.solv = None
        self.model = None

        # Search params
        self.max_searches = max_searches
        self.ws_range = [-1, 1]
        self.lr_range = [-1, 1]     # TODO : fix these up
        self.reg_range = [-1, 1]

        # Solver params


        # Other params
        self.verbose

    def load_data(self, data_dir):
        self.dataset = data_utils.get_CIFAR10_data(data_dir)

        if self.verbose:
            for k, v in self.dataset.items():
                print("%s : %s " % (k, v.shape))


    def param_search(self):
        # TODO : store the range in self (eg: to be able to launch many
        # instances


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
                                              optim_config={'learning_rate': learning_rate},
                                              verbose=self.verbose)
            if self.verbose:
                print(self.model)
            self.solv = solver.Solver(self.model,
                                      self.train_data,
                                      num_epochs=self.solver_num_epochs,
                                      batch_size=self.solver_batch_size,
                                      update_rule=self.solver_update_rule,
                                      verbose=self.verbose,
                                      print_every=self.solver_print_every
                                      )
            self.solv.train()
            n += 1




