"""
CONV_PARAM_SEARCH
Find suitable hyperparameters for a convnet

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local modules
import matplotlib.pyplot as plt
import numpy as np
import pymllib.classifiers.convnet as convnet
import pymllib.solver.solver as solver
import pymllib.utils.data_utils as data_utils
import pymllib.vis.vis_weights as vis_weights

def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

def find_params(data, checkpoint_name=None, verbose=True):

    input_dim = (3, 32, 32)
    num_epochs = 30

    param_search = True
    num_searches = 0
    while param_search:
        weight_scale = 10 ** (np.random.uniform(-6, -1))
        learning_rate = 10 ** (np.random.uniform(-4, -1))
        model = convnet.ConvNetLayer(input_dim=input_dim,
                        hidden_dims=[256, 100],
                        num_filters = [16, 32],
                        weight_scale=weight_scale,
                        dtype=np.float32)
        if verbose:
            print(model)
        model_solver = solver.Solver(model,
                                    data,
                                    print_every=10,
                                    num_epochs=num_epochs,
                                    batch_size=50,     # previously 25
                                    update_rule='adam',
                                    optim_config={'learning_rate': learning_rate},
                                    checkpoint_name=checkpoint_name,
                                    checkpoint_dir='examples')
        model_solver.train()
        num_searches += 1
        if max(model_solver.train_acc_history) >= 1.0:
            param_search = False
            lr = learning_rate
            ws = weight_scale
            print("Found parameters after %d epochs total (%d searches of %d epochs each)" % (num_searches * num_epochs, num_searches, num_epochs))

    print("Best learning rate is %f" % lr)
    print("Best weight scale is %f" % ws)


if __name__ == "__main__":

    verbose = True
    data_dir = 'datasets/cifar-10-batches-py'
    dataset = load_data(data_dir, verbose)
    num_train = 250

    train_data = {
        'X_train': dataset['X_train'][:num_train],
        'y_train': dataset['y_train'][:num_train],
        'X_val': dataset['X_val'][:num_train],
        'y_val': dataset['y_val'][:num_train]
    }

    #conv4_fc2_net = convnet.ConvNetLayer()         #TODO : layers
    #conv4_fc2_solver = solver.Solver(conv4_fc2_net, dataset)

    cname = 'conv2-fc2-param-search.pkl'
    find_params(train_data, checkpoint_name=cname)
