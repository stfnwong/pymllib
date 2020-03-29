"""
OPTIM

Optimnization routines for solver
TODO : Docstrings
"""

import numpy as np
from typing import Any
from typing import Dict
from typing import Tuple


def sgd(w:np.ndarray, dw:np.ndarray, config:Dict[str, Any]=None) -> Tuple[np.ndarray, Dict[str, Any]]:
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


def sgd_momentum(w:np.ndarray, dw:np.ndarray, config:Dict[str, Any]=None) -> Tuple[str, Dict[str,Any]]:
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


def rmsprop(x:np.ndarray, dx:np.ndarray, config:Dict[str, Any]=None) -> Tuple[np.ndarray, Dict[str, Any]]:
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


def adam(x:np.ndarray, dx:np.ndarray, config:Dict[str, Any]=None) -> Tuple[np.ndarray, Dict[str, Any]]:

    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('epsilon', 1e-8)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('t', 0)

    lr = config['learning_rate']
    b1 = config['beta1']
    b2 = config['beta2']
    eps = config['epsilon']

    config['t'] += 1
    config['m'] = b1 * config['m'] + (1 - b1) * dx
    config['v'] = b2 * config['v'] + (1 - b2) * (dx**2)
    mt_hat = config['m'] / (1 - (b1)**config['t'])
    vt_hat = config['v'] / (1 - (b2)**config['t'])
    next_x = x - lr * mt_hat / (np.sqrt(vt_hat + eps))

    return next_x, config
