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

def rmsprop(x, dx, config=None):

    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    #mnsqr = config['cache']
    config['cache'] = config['cache'] * config['decay_rate'] + (1 - config['decay_rate']) * dx**2
    next_x = x - config['learning_rate'] * dx / np.sqrt(config['cache'] + config['epsilon'])
    #config['cache'] = config['cache']

    return next_x, config

def adam(x, dx, config=None):
    pass




    #next_x = x - config['learning_rate'] * dx / (np.sqrt(mnsqr + config['epsilon'])

