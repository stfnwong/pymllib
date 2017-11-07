"""
TEST_AUTOENCODER

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import matplotlib.pyplot as plt
import numpy as np
import unittest

import data_utils
import autoencoder
import solver


def load_data(data_dir, verbose=False):

    if verbose is True:
        print("Loading data from %s" % data_dir)

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

def get_figure_handles():
    fig = plt.figure()
    ax = []
    for i in range(3):
        sub_ax = fig.add_subplot(3,1,(i+1))
        ax.append(sub_ax)

    return fig, ax

# Show the solver output
def plot_test_result(ax, solver_dict, num_epochs=None):

    assert len(ax) == 3, "Need 3 axis"

    for n in range(len(ax)):
        ax[n].set_xlabel("Epoch")
        if num_epochs is not None:
            ax[n].set_xticks(range(num_epochs))
        if n == 0:
            ax[n].set_title("Training Loss")
        elif n == 1:
            ax[n].set_title("Training Accuracy")
        elif n == 2:
            ax[n].set_title("Validation Accuracy")

    # update data
    for method, solv in solver_dict.items():
        ax[0].plot(solv.loss_history, 'o', label=method)
        ax[1].plot(solv.train_acc_history, '-x', label=method)
        ax[2].plot(solv.val_acc_history, '-x', label=method)

    # Update legend
    for i in range(len(ax)):
        ax[i].legend(loc='upper right', ncol=4)

    # Note: outside the function we set
    # fig.set_size_inches(8,8)
    # fig.tight_layout()


class TestAutoencoder(unittest.TestCase):
    """
    TestAutoencoder
    """

    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.verbose = True
        self.eps = 1e-6

    def test_autoencoder_loss(self):
        print("\n======== TestAutoencoder.test_autoencoder_loss:")
        dataset = load_data(self.data_dir, self.verbose)
        num_train = 100

        small_data = {
            'X_train': dataset['X_train'][:num_train],
            'y_train': dataset['X_train'][:num_train],
            'X_val':   dataset['X_val'][:num_train],
            'y_val':   dataset['X_val'][:num_train]
        }
        # Reshape some of the data in the small training set
        y_train_shape = (small_data['y_train'].shape[0], np.prod(small_data['y_train'].shape[1:]))
        y_val_shape = (small_data['y_val'].shape[0], np.prod(small_data['y_val'].shape[1:]))
        small_data['y_train'] = np.reshape(small_data['y_train'], y_train_shape)
        small_data['y_val'] = np.reshape(small_data['y_val'], y_val_shape)

        if self.verbose:
            print("Small data set")
            for k, v in small_data.items():
                print("%s : %s : " % (k, v.shape))

        #input_dim = small_data['X_train'].shape[0]
        input_dim = 3 * 32 * 32
        hidden_dims = [int(input_dim / 4)]
        weight_scale = 0.079564
        learning_rate = 0.003775        # experimentally derived
        reg = 1.0

        # Get an autoencoder
        auto_model = autoencoder.Autoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            reg=reg,
            weight_scale=weight_scale,
            verbose=self.verbose)

        if self.verbose:
            print(auto_model)
        # Get a solver
        auto_solver = solver.Solver(auto_model,
                                    small_data,
                                    print_every=10,
                                    num_epochs=30,
                                    batch_size=50,     # previously 25
                                    update_rule='sgd',
                                    optim_config={'learning_rate': learning_rate})
        auto_solver.train()

        print("======== TestAutoencoder.test_autoencoder_loss: <END> ")



if __name__ == "__main__":
    unittest.main()
