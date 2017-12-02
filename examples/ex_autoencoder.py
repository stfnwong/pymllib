"""
AUTOENCODER
An example of training with an autoencoder

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Autoencoder
from pymllib.classifiers import autoencoder
from pymllib.utils import data_utils
from pymllib.solver import solver

# Debug
from pudb import set_trace; set_trace()

# Global for where dataset is kept
data_dir = 'datasets/cifar-10-batches-py'

def run_autoencoder(verbose=True):
    dataset = data_utils.get_CIFAR10_data(data_dir)

    if verbose:
        print('Items in dataset:')
        for k, v in dataset.items():
            print('%s : %s' % (k, v.shape))

    # Model hyperparameters
    hidden_dims = 256
    input_dim = 3 * 32 * 32
    dropout = 0
    reg = 0.0
    weight_scale = 1e-2
    seed = None
    # Solver hyperparameters
    learning_rate=1e-3
    batch_size=64
    update_rule='adam'
    # Get a model
    model = autoencoder.Autoencoder(hidden_dims=hidden_dims,
                                    input_dim=input_dim,
                                    drouput=dropout,
                                    reg=reg,
                                    weight_scale=weight_scale,
                                    seed=seed,
                                    verbose=verbose)
    if verbose:
        print(model)

    # Get a solver
    solv = solver.Solver(model, dataset,
                         optim_config={'learning_rate': learning_rate},
                         update_rule=update_rule,
                         verbose=verbose,
                         batch_size=batch_size,
                         num_epochs=20)
    solv.train()

    # Display results ?


if __name__ == '__main__':
    run_autoencoder()
