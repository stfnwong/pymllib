"""
CONVNET_UTILS

Stefan Wong 2017
"""

import numpy as np

def get_conv_layer_dict(model, verbose=False):

    conv_layers = {}
    for k, v in model.params.items():
        if k[:1] == 'W':
            if len(model.params[k].shape) == 4:
                conv_layers[k] = model.params[k]

    if verbose is True:
        for k, v in conv_layers.items():
            print("%s : %s" % (k, v.shape))

    return conv_layers

def print_conv_sizes(param_dict):

    for k, v in param_dict.items():
        if k[:1] == 'W':
            if len(param_dict[k].shape) == 4:
                print("%s : %s " % (k, v.shape))



# Generate random data for testing grad descent
def gen_random_data(data, data_scale=256):

    rand_data = {}
    for k, v in data.items():
        rand_data[k] = data_scale * np.random.randn(v.shape)

    return rand_data
