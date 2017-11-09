"""
CONVNET_UTILS

Stefan Wong 2017
"""

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
