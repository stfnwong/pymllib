"""
EX_XAVIER_TRAIN
Example using a convnet with xavier parameters

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from pymllib.classifiers import convnet
from pymllib.solver import solver
from pymllib.vis import vis_solver
from pymllib.utils import data_utils

# Debug
from pudb import set_trace; set_trace()


def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset


def train_xavier(verbose=True, draw_plots=False):

    data_dir = 'datasets/cifar-10-batches-py'
    dataset = load_data(data_dir)

    # Hyperparams
    input_dim = (3, 32, 32)
    hidden_dims = [256, 256]
    num_filters = [16, 32, 64]
    reg = 2e-2
    weight_scale = 1e-3
    learning_rate = 1e-3
    num_epochs = 600
    batch_size = 50
    update_rule='adam'

    weight_init = ['gauss', 'gauss_sqrt', 'xavier']
    model_dict = {}

    for w in weight_init:
        model = convnet.ConvNetLayer(input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        num_filters = num_filters,
                        weight_scale=weight_scale,
                        weight_init=w,
                        reg=reg,
                        verbose=True)
        model_dict[w] = model

    solver_dict = {}

    for k, m in model_dict.items():
        if verbose:
            print(m)
        solv = solver.Solver(m,
                             dataset,
                             print_every=10,
                             num_epochs=num_epochs,
                             batch_size=batch_size,
                             update_rule=update_rule,
                             optim_config={'learning_rate': learning_rate})
        solv.train()
        fname = '%s-solver-%d-epochs.pkl' % (k, int(num_epochs))
        solv.save(fname)
        skey = '%s-%s' % (m.__repr__(), k)
        solver_dict[skey] = solv

    # Plot results
    if draw_plots is True:
        fig, ax = vis_solver.get_train_fig()
        vis_solver.plot_solver_compare(ax, solver_dict)
        plt.show()


if __name__ == "__main__":
    train_xavier(True, False)
