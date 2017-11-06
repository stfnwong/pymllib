"""
TEST_AUTOENCODER
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import matplotlib.pyplot as plt
import numpy as np
import data_utils
import check_gradient
import error
import layers
import autoencoder
import twolayer_modular as twol
import solver

import unittest
# Debug
from pudb import set_trace; set_trace()

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


class TestAutoencoder(unittest.TestCase):

    def test_autoencoder(self):

        input_dim = 3 * 32 * 33
        hidden_dims = [int(input_dim / 8)]

        auto = autoencoder.Autoencoder()



if __name__ == "__main__":
    unittest.main()
