"""
AUTO_TEST_
A new autoencoder implementation test

Stefan Wong 2018
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pymllib.layers.layers as layers

# Debug
from pudb import set_trace; set_trace()


class AutoTest(object):
    def __init__(self, **kwargs):
        self.verbose = kwargs.pop('verbose', False)

        # Init the internal params
        self.params = dict()
        # because this is just a test, we haved a fixed network layout

    def __repr__(self):
        return 'auto_test'

    # For test purposes, lets just create an autoencoder of
    # fixed size, get that working, then try to generalize
    def loss(self, X, y=None):
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        y = np.reshape(y, (y.shape[0], np.prod(y.shape[1:]))).T
