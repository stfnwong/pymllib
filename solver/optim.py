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
    w -= config['learning_rate'] * dw

    return w, config

