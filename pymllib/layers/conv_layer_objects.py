"""
CNN_LAYER_OBJECTS
Object-oriented implementation of CNN layers and
combination layers.

Stefan Wong 2017
"""


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


class CNNLayer(object):
    def __init__(self, N, C, H, W):
        self.W = np.random.randn((N, C, H, W))
        self.b = np.zeros(N)

    def update(self, next_w, next_b):
        self.W = next_w
        self.b = next_b


class ConvLayer(CNNLayer):
    def __str__(self):
        s = []
        s.append('CNN Layer:\n')
        s.append('Shape : %s' % str(self.W.shape))
        return ''.join(s)

    # TODO
    def forward(self):
        pass

    def backward(self):
        pass

