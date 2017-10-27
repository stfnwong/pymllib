"""
OPTIM

Optimnization routines for solver
TODO : Docstrings
"""

import numpy as np

def sgd(w, dw, config=None):
    """
    Perform vanilla stochastic gradient descent

    config format:
        - learning_rate : Scalar learning rate
    """

    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    next_w = w - config['learning_rate'] * dw

    return next_w, config

def sgd_momentum(w, dw, config=None):
    """
    Perform stochastic gradient descent with momentum

    """

    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.7)
    v = config.get('velocity', np.zeros_like(w))
    next_v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + next_v
    config['velocity'] = next_v

    return next_w, config

# TODO : this is just here to shut the linter up
def rmsprop(w, dw, config=None):
    pass
