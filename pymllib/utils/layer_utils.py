"""
LAYER_UTILS

Stefan Wong 2017
"""

import numpy as np

# list of available weight init methods
valid_init_methods = ['gauss', 'gauss_sqrt', 'gauss_sqrt2', 'xavier']

def fc_layer_weight_init(weight_scale:float, weight_init:str, N:int, D:int) -> np.ndarray:
    """
    FC_LAYER_WEIGHT_INIT

    Init weights for a given fully-connected layer using the specified
    init method.
    """

    if weight_init == 'gauss':
        W = weight_scale * np.random.randn(N, D)
    elif weight_init == 'gauss_sqrt':
        W = weight_scale * np.random.randn(N, D) * (1 / np.sqrt(2.0 / (N+D)))
    elif weight_init == 'gauss_sqrt2':
        W = np.random.randn(N, D) * (1 / np.sqrt(2/(N+D)))
    elif weight_init == 'xavier':
        w_lim = 2 / np.sqrt(N + D)
        W = np.random.uniform(low=-w_lim, high=w_lim, size=(N, D))
    else:
        raise ValueError('Invalid weight init method %s' % weight_init)

    return W


def conv_layer_weight_init(weight_scale:float, weight_init:str, N:int, D:int, f:int) -> np.ndarray:
    """
    CONV_LAYER_WEIGHT_INIT

    Init weights for a given convolutional layer using the specified
    init method.
    """

    if weight_init == 'gauss':
        W = weight_scale * np.random.randn(N, D, f)
    elif weight_init == 'gauss_sqrt':
        W = weight_scale * np.random.randn(N, D, f, f) * (1 / np.sqrt(2.0 / (N+D)))
    elif weight_init == 'gauss_sqrt2':
        W = np.random.randn(N, D, f, f) * (1 / np.sqrt(2/(N+D)))
    elif weight_init == 'xavier':
        w_lim = 2 / np.sqrt(N + D, f, f)
        W = np.random.uniform(low=-w_lim, high=w_lim, size=(N, D, f, f))
    else:
        raise ValueError('Invalid weight init method %s' % weight_init)

    return W
