"""
TEST_FCNET_OBJECT

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
import unittest

# Modules
from pymllib.classifiers import fcnet_object
#from pymllib.solver import solver_object    # TODO
from pymllib.utils import data_utils
from pymllib.vis import vis_solver

# Debug
#from pudb import set_trace; set_trace()


def load_data(data_dir, verbose=False):

    if verbose is True:
        print("Loading data from %s" % data_dir)

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset



class TestFCNetObject(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'datasets/cifar-10-batches-py'
        self.verbose = True

    def test_layer_creation(self):

        print("\n======== TestFCNetObject.test_layer_creation:")

        hidden_dims = [100, 100, 100]
        fcnet = fcnet_object.FCNetObject(hidden_dims)

        print(fcnet)



        print("======== TestFCNetObject.test_layer_creation: <END> ")


if __name__ == '__main__':
    unittest.main()
