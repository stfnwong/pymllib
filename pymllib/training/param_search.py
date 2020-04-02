"""
CONV_PARAM_SEARCH
Find suitable parameters for convolutional neural network

Stefan Wong 2017
"""

import numpy as np
# Lib
from pymllib.solver import solver
from pymllib.classifiers import convnet
from pymllib.utils import data_utils

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union


# Debug
#from pudb import set_trace; set_trace()

class ConvParamSearch:
    def __init__(self, data:Any, model:Any, **kwargs) -> None:
        # Reserve names for model and solver

        # Search params
        self.max_searches :int             = kwargs.pop('max_searches', 1e3)
        self.num_train    :int             = kwargs.pop('num_train', 500)
        self.ws_range     :Tuple[int, int] = kwargs.pop('ws_range', [-6, 1])
        self.lr_range     :Tuple[int, int] = kwargs.pop('lr_range', [-5, 1])
        self.reg_range    :Tuple[int, int] = kwargs.pop('reg_range', [-3, 1])
        # Model params
        self.model_input_dim     :Tuple[int, int, int] = kwargs.pop('input_dim', (3, 32, 32))
        self.model_hidden_dims   :List[int] = kwargs.pop('hidden_dims', [256, 256])
        self.model_num_filters   :List[int] = kwargs.pop('num_filters', [16, 32, 64, 128])
        self.model_num_classes   :int       = kwargs.pop('num_classes', 10)
        self.model_use_batchnorm :bool      = kwargs.pop('use_batchnorm', True)
        # Solver params
        self.solver_num_epochs      = kwargs.pop('num_epochs', 100)
        self.solver_batch_size      = kwargs.pop('batch_size', 20)
        self.solver_update_rule     = kwargs.pop('update_rule', 'adam')
        self.solver_checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.solver_checkpoint_dir  = kwargs.pop('checkpoint_dir', '.')
        self.solver_print_every     = kwargs.pop('print_every', 20)
        # Output params
        self.lr_output  :float = 0.0
        self.ws_output  :float = 0.0
        self.reg_output :float = 0.0
        # Other params
        self.verbose    :bool  = kwargs.pop('verbose', False)

        # Internal params
        self.solv = None
        self.train_data = None
        if data is None or model is None:
            self.model = None
            self.dataset :Union[None, Dict[str, Any]] = None

            return

        # TODO : Split as per solver?
        self.model:Any = model
        self.data :Dict[str, Any] = data

    def __str__(self) -> str:
        s = []
        s.append("Model parameters\n")

        if self.model is None:
            # Make a model to simplify printing
            model = convnet.ConvNetLayer(input_dim=self.model_input_dim,
                                         hidden_dims=self.model_hidden_dims,
                                         num_filters=self.model_num_filters)
        else:
            model = self.model

        s.append(str(model))
        s.append("\n")
        s.append("Solver parameters\n")

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
        s.append("\n")

        return ''.join(s)

    def __repr__(self) -> str:
        return self.__str__()

    def load_data(self, data_dir:str) -> None:
        self.dataset = data_utils.get_CIFAR10_data(data_dir)
        self.train_data :Dict[str, Any] = {
            'X_train': self.dataset['X_train'][:self.num_train],
            'y_train': self.dataset['y_train'][:self.num_train],
            'X_val':   self.dataset['X_val'][:self.num_train],
            'y_val':   self.dataset['y_val'][:self.num_train]
        }

        if self.verbose:
            for k, v in self.dataset.items():
                print("%s : %s " % (k, v.shape))

    def init_model(self, weight_scale:float, reg:float) -> None:
        self.model = convnet.ConvNetLayer(input_dim=self.model_input_dim,
                                            hidden_dims=self.model_hidden_dims,
                                            num_filters=self.model_num_filters,
                                            use_batchnorm=self.model_use_batchnorm,
                                            reg=reg,
                                            weight_scale=weight_scale,
                                            verbose=self.verbose)

    def init_solver(self, data:Dict[str, Any], learning_rate:float=1e-3, num_epochs:Union[None, int]=None) -> None:
        if num_epochs is None:
            num_epochs = self.solver_num_epochs
        self.solv = solver.Solver(self.model,
                                    data,
                                    num_epochs=num_epochs,
                                    batch_size=self.solver_batch_size,
                                    update_rule=self.solver_update_rule,
                                    optim_config={'learning_rate': learning_rate},
                                    verbose=self.verbose,
                                    print_every=self.solver_print_every,
                                    checkpoint_name=self.solver_checkpoint_name,
                                    checkpoint_dir=self.solver_checkpoint_dir)


    def overfit_data(self, overfit_sizes=[10, 50, 100, 200], learning_rate=1e-3):
        """
        Overfit test. Attempt to overfit the model on a small dataset
        """

        # Init the model here if it has been set to None
        if self.model is None:
            if self.verbose:
                print("Model is not initialized, exiting overfit test\n")
            return

        if self.verbose:
            print("Ovefitting on sizes %s" % str(overfit_sizes))

        for size in overfit_sizes:
            overfit_data = {
                'X_train': self.dataset['X_train'][:size],
                'y_train': self.dataset['y_train'][:size],
                'X_val':   self.dataset['X_val'][:size],
                'y_val':   self.dataset['y_val'][:size]
            }
            self.init_solver(overfit_data, learning_rate=learning_rate)
            if self.verbose:
                print("Attemping to overfit on %d examples" % size)
            self.solv.train()
            if max(self.solv.train_acc_history) < 1.0:
                print("Failed to overfit on dataset : \n")
                for k, v in overfit_data.items():
                    print("%s : %s " % (k, v.shape))
                # TODO : transform to an exception?
                return -1

        return None

    def param_search(self):
        """
        PARAM_SEARCH

        Attempt to find parameters that allow the network to train
        """

        param_search = True
        n = 0
        while param_search and n < self.max_searches:
            weight_scale = 10 ** (np.random.uniform(self.ws_range[0], self.ws_range[1]))
            learning_rate = 10 ** (np.random.uniform(self.lr_range[0], self.lr_range[1]))
            reg = 10**(np.random.uniform(self.reg_range[0], self.reg_range[1]))

            if self.verbose:
                print("Selected weight scale  = %f" % (weight_scale))
                print("Selected learning rate = %f" % (learning_rate))
                print("Selected reg strength  = %f" % (reg))

            # Get a model
            self.init_model(weight_scale=weight_scale, reg=reg)
            if self.verbose:
                print(self.model)

            # Attempt to overfit some small data with these parameters
            rv = self.overfit_data(learning_rate=learning_rate)
            if rv is not None:
                if self.verbose:
                    print("Failed to overfit with lr=%f, ws=%f, reg=%f\n" % (learning_rate, weight_scale, reg))
                continue

            if self.verbose:
                print('Training with lr = %f, ws = %f, reg = %f' % (learning_rate, weight_scale, reg))
            if self.verbose and self.solver_checkpoint_name is not None:
                print("Saving solver checkpoints to file %s/%s" % (self.solver_checkpoint_dir, self.solver_checkpoint_name))

            #Get a solver
            self.init_solver(self.train_data, learning_rate=learning_rate)
            self.solv.train()
            n += 1
            # Found correct params
            if max(self.solv.train_acc_history) >= 1.0:
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
