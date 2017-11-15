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
        self.max_searches = kwargs.pop('max_searches', 1e3)
        self.num_train = kwargs.pop('num_train', 500)
        self.ws_range = kwargs.pop('ws_range', [-6, 1])
        self.lr_range = kwargs.pop('lr_range', [-5, 1])
        self.reg_range = kwargs.pop('reg_range', [-3, 1])
        # Model params
        self.model_input_dim = kwargs.pop('input_dim', (3, 32, 32))
        self.model_hidden_dims = kwargs.pop('hidden_dims', [256, 256])
        self.model_num_filters = kwargs.pop('num_filters', [16, 32, 64, 128])
        self.model_num_classes = kwargs.pop('num_classes', 10)
        self.model_use_batchnorm = kwargs.pop('use_batchnorm', True)
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

    def init_model(self, weight_scale, reg):
        self.model = convnet.ConvNetLayer(input_dim=self.model_input_dim,
                                            hidden_dims=self.model_hidden_dims,
                                            num_filters=self.model_num_filters,
                                            use_batchnorm=self.model_use_batchnorm,
                                            reg=reg,
                                            weight_scale=weight_scale,
                                            verbose=self.verbose)

    def init_solver(self, data, learning_rate=1e-3, num_epochs=None):

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


def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset


def convert_data_random(data, data_scale=256):

    rand_data = {}
    for k, v in data.items():
        #rand_data[k] = data_scale * np.random.randn(v.shape)
        rand_data[k] = np.random.random_integers(0, data_scale, v.shape)

    return rand_data

def gen_random_data(num_train=8000, num_val=800, num_test=800, num_classes=10, data_scale=256):

    data = {'X_train': np.random.randn(num_train, 3, 32, 32),
            'y_train': np.random.random_integers(0, num_classes-1, size=num_train),
            'X_val': np.random.randn(num_val, 3, 32, 32),
            'y_val': np.random.random_integers(0, num_classes-1, num_val),
            'X_test': np.random.randn(num_test, 3, 32, 32),
            'y_test': np.random.random_integers(0, num_classes-1, num_test)
            }

    return data

def scale_network():

    # Some trial hyperparameters
    reg = 1e-4
    ws = 0.05
    lr = 1e-3
    fsizes = [16, 32, 64, 128]
    hdims = 256

    num_filters = []
    hidden_dims = [256]
    num_epochs = 50

    # prep data
    num_train = 5000
    dataset = load_data('datasets/cifar-10-batches-py')
    small_data = {
        'X_train': dataset['X_train'][:num_train],
        'y_train': dataset['y_train'][:num_train],
        'X_val':   dataset['X_val'][:num_train],
        'y_val':   dataset['y_val'][:num_train]
    }

    for s in fsizes:
        num_filters.append(s)
        model = convnet.ConvNetLayer(hidden_dims=hidden_dims,
                                     num_filters=num_filters,
                                     reg=reg,
                                     weight_scale=ws,
                                     verbose=True)
        print(model)
        solv = solver.Solver(model, small_data,
                             optim_config={'learning_rate': lr},
                             update_rule='sgd_momentum',
                             num_epochs=num_epochs,
                             batch_size=50,
                             loss_window_len=250)
        solv.train()

        # Show results
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = []
        for i in range(3):
            subax = fig.add_subplot(3, 1, (i+1))
            ax.append(subax)

        ax[0].plot(solv.loss_history, 'o')
        ax[0].set_title("Loss")
        ax[1].plot(solv.train_acc_history)
        ax[1].set_title("Training accuracy")
        ax[2].plot(solv.val_acc_history)
        ax[2].set_title("Validation accuracy")

        for i in range(3):
            ax[i].set_xlabel("Epochs")
            #ax[i].set_xticks(range(num_epochs))
        plt.show()

def learn_random_data():
    # Some trial hyperparameters
    reg = 1e-4
    ws = 0.05
    lr = 1e-3
    num_epochs = 10

    #data = load_data('datasets/cifar-10-batches-py', verbose=True)
    #rand_data = convert_data_random(data, int(np.max(data['X_train'])))
    rand_data = gen_random_data()
    # Get model
    model = convnet.ConvNetLayer(hidden_dims=[256],
                                 reg=reg)
    # Get solver
    solv = solver.Solver(model,
                         rand_data,
                         optim_config={'learning_rate': lr},
                         num_epochs=num_epochs)
    solv.train()

    # Show some plots
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = []
    for i in range(3):
        subax = fig.add_subplot(3, 1, (i+1))
        ax.append(subax)

    ax[0].plot(solv.loss_history, 'o')
    ax[0].set_title("Loss")
    ax[1].plot(solv.train_acc_history)
    ax[1].set_title("Training accuracy")
    ax[2].plot(solv.val_acc_history)
    ax[2].set_title("Validation accuracy")

    for i in range(3):
        ax[i].set_xlabel("Epochs")
        ax[i].set_xticks(range(num_epochs))





# Basic test
if __name__ == "__main__":
    scale_network()
    #data_dir = 'datasets/cifar-10-batches-py'
    #searcher = ConvParamSearch(lr_range=[-6, -3],
    #                           ws_range=[-5, -1],
    #                           reg_range=[-3, -1],
    #                           checkpoint_name='c4fc2',
    #                           checkpoint_dir='examples',
    #                           num_train=10000,
    #                           num_epochs=500,
    #                           batch_size=100,
    #                           verbose=True)
    ##print(searcher)     # TODO : Fix all the __str__ methods
    #searcher.load_data(data_dir)
    #searcher.param_search()
