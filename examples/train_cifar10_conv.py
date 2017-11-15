"""
TRAIN_CIFAR10_CONV

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


def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset


def train_cifar10_conv():
    data_dir = 'datasets/cifar-10-batches-py'
    data = load_data(data_dir)

    verbose = True
    # Model hyperparams
    weight_scale = 0.05
    filter_size = 3
    reg = 0.05
    input_dim = (3, 32, 32)
    num_filters = [16, 32, 64, 128]
    hidden_dims = [256, 256]
    # Solver hyperparams
    update_rule = 'sgd_momentum'
    learning_rate = 1e-3
    num_epochs = 2000

    # Get a model
    conv_model = convnet.ConvNetLayer(input_dim=input_dim,
                                      hidden_dims=hidden_dims,
                                      num_filters=num_filters,
                                      weight_scale=weight_scale,
                                      reg=reg,
                                      filter_size=filter_size,
                                      verbose=verbose)
    if verbose:
        print(conv_model)
    # Get a solver
    checkpoint_name='c4-16-32-64-128-f2-256-256-lr=%f-ws=%f' % (learning_rate, weight_scale)
    conv_solver = solver.Solver(conv_model, data,
                                num_epochs=num_epochs,
                                batch_size=50,
                                update_rule=update_rule,
                                optim_config={'learning_rate': learning_rate},
                                verbose=verbose,
                                print_every=50,
                                checkpoint_name=checkpoint_name,
                                checkpoint_dir='examples')
    if verbose is True:
        print("Training %d layer net" % conv_model.num_layers)
    conv_solver.train()

if __name__ == "__main__":
    train_cifar10_conv()
