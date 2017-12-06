"""
RNN UTILS
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymllib.layers import rnn_layers

import numpy as np

def check_loss(N, T, V, p, verbose=False):
    x = 0.001 * np.random.randn(N, T, V)
    y = np.random.randint(V, size=(N, T))
    mask = np.random.randn(N, T) <= p

    out = rnn_layers.temporal_softmax_loss(x, y, mask)[0]
    if verbose:
        print(out)

    return out
