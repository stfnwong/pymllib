"""
EX_OVERFIT
Try to overfit progressively larger amounts of 'small' data

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt

# Local libs
from pymllib.solver import solver
from pymllib.classifiers import convnet
from pymllib.utils import data_utils
from pymllib.vis import vis_solver

# Debug
from pudb import set_trace; set_trace()


def overfit():
    # Data
    dataset = data_utils.get_CIFAR10_data('datasets/cifar-10-batches-py')
    # Hyperparameters
    # for now we just some random params, not found by search
    reg = 1e-2
    weight_scale = 2e-3
    learning_rate = 1e-3
    # Training parameters
    num_epochs = 40
    #train_sizes = [50, 100, 150, 200]
    train_sizes = [200, 400, 800, 1000, 1500]

    solv_dict = {}
    for size in train_sizes:
        overfit_data = {
            'X_train': dataset['X_train'][:size],
            'y_train': dataset['y_train'][:size],
            'X_val':   dataset['X_val'][:size],
            'y_val':   dataset['y_val'][:size]
        }
        model = convnet.ConvNetLayer(hidden_dims=[256],
                                     num_filters=[16],
                                     filter_size=5,
                                     reg=reg,
                                     weight_scale=weight_scale)
        solv = solver.Solver(model,
                             overfit_data,
                             num_epochs=num_epochs,
                             optim_config={'learning_rate': learning_rate})
        print("Overfitting on %d examples in %d epochs using the following network" % (size, num_epochs))
        print(model)
        solv.train()
        dkey = 'size_%d' % size
        solv_dict[dkey] = solv
        # Check that we can actually overfit

    # Plot the results
    fig, ax = vis_solver.get_train_fig()
    vis_solver.plot_solver_compare(ax, solv_dict)
    plt.show()


if __name__ == '__main__':
    overfit()
