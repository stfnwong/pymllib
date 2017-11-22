"""
CONV_PARAM_SEARCH
Find suitable parameters for convolutional neural network

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# Lib
from pymllib.solver import solver
from pymllib.classifiers import convnet
from pymllib.utils import data_utils
from pymllib.training import param_search

# Debug
from pudb import set_trace; set_trace()


def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset


def convert_data_random(data, data_scale=256):

    rand_data = {}
    for k, v in data.items():
        #rand_data[k] = data_scale * np.random.randn(v.shape)
        rand_data[k] = np.random.random_integers(0, data_scale, v.shape)

    return rand_data

def gen_random_data(num_train=8000, num_val=800, num_test=800, num_classes=10, data_scale=256):

    data = {'X_train': np.random.randn(num_train, 3, 32, 32),
            'y_train': np.random.random_integers(0, num_classes-1, size=num_train),
            'X_val': np.random.randn(num_val, 3, 32, 32),
            'y_val': np.random.random_integers(0, num_classes-1, num_val),
            'X_test': np.random.randn(num_test, 3, 32, 32),
            'y_test': np.random.random_integers(0, num_classes-1, num_test)
            }

    return data

def scale_network(draw_plots=False):

    # Some trial hyperparameters
    reg = 1e-4
    ws = 0.05
    lr = 1e-3
    fsizes = [16, 32, 64, 128]
    hdims = 256

    num_filters = []
    hidden_dims = [256]
    num_epochs = 100

    # prep data
    num_train = 5000
    dataset = load_data('datasets/cifar-10-batches-py')
    small_data = {
        'X_train': dataset['X_train'][:num_train],
        'y_train': dataset['y_train'][:num_train],
        'X_val':   dataset['X_val'][:num_train],
        'y_val':   dataset['y_val'][:num_train]
    }

    for s in fsizes:
        num_filters.append(s)
        if s == 64:
            hidden_dims.append(hdims)
        model = convnet.ConvNetLayer(hidden_dims=hidden_dims,
                                     num_filters=num_filters,
                                     reg=reg,
                                     weight_scale=ws,
                                     verbose=True)
        print(model)
        cname = model.__repr__()
        print("Saving checkpoints to examples/%s.pkl" % cname)
        solv = solver.Solver(model, small_data,
                             optim_config={'learning_rate': lr},
                             update_rule='sgd_momentum',
                             num_epochs=num_epochs,
                             checkpoint_dir='examples',
                             checkpoint_name=cname,
                             batch_size=50,
                             loss_window_len=400,
                             loss_window_eps=1e-5)
        solv.train()

        # Show results
        if draw_plots is True:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = []
            for i in range(3):
                subax = fig.add_subplot(3, 1, (i+1))
                ax.append(subax)

            ax[0].plot(solv.loss_history, 'o')
            ax[0].set_title("Loss")
            ax[1].plot(solv.train_acc_history)
            ax[1].set_title("Training accuracy")
            ax[2].plot(solv.val_acc_history)
            ax[2].set_title("Validation accuracy")

            for i in range(3):
                ax[i].set_xlabel("Epochs")
                #ax[i].set_xticks(range(num_epochs))
            plt.show()

def learn_random_data():
    # Some trial hyperparameters
    reg = 1e-4
    ws = 0.05
    lr = 1e-3
    num_epochs = 10

    #data = load_data('datasets/cifar-10-batches-py', verbose=True)
    #rand_data = convert_data_random(data, int(np.max(data['X_train'])))
    rand_data = gen_random_data()
    # Get model
    model = convnet.ConvNetLayer(hidden_dims=[256],
                                 reg=reg)
    # Get solver
    solv = solver.Solver(model,
                         rand_data,
                         optim_config={'learning_rate': lr},
                         num_epochs=num_epochs)
    solv.train()

    # Show some plots
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = []
    for i in range(3):
        subax = fig.add_subplot(3, 1, (i+1))
        ax.append(subax)

    ax[0].plot(solv.loss_history, 'o')
    ax[0].set_title("Loss")
    ax[1].plot(solv.train_acc_history)
    ax[1].set_title("Training accuracy")
    ax[2].plot(solv.val_acc_history)
    ax[2].set_title("Validation accuracy")

    for i in range(3):
        ax[i].set_xlabel("Epochs")
        ax[i].set_xticks(range(num_epochs))





# Basic test
if __name__ == "__main__":
    scale_network()
    #data_dir = 'datasets/cifar-10-batches-py'
    #searcher = ConvParamSearch(lr_range=[-6, -3],
    #                           ws_range=[-5, -1],
    #                           reg_range=[-3, -1],
    #                           checkpoint_name='c4fc2',
    #                           checkpoint_dir='examples',
    #                           num_train=10000,
    #                           num_epochs=500,
    #                           batch_size=100,
    #                           verbose=True)
    ##print(searcher)     # TODO : Fix all the __str__ methods
    #searcher.load_data(data_dir)
    #searcher.param_search()
