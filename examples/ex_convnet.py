"""
<<<<<<< HEAD
X_CONVNET
=======
EX_CONVNET
>>>>>>> layer-object
Some examples of things that we can do with convnets

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import pymllib.classifiers.convnet as convnet
import pymllib.solver.solver as solver
import pymllib.utils.data_utils as data_utils
import pymllib.vis.vis_weights as vis_weights

# Debug
from pudb import set_trace; set_trace()

def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

def get_figure_handles():
    fig = plt.figure()
    ax = []
    for i in range(3):
        sub_ax = fig.add_subplot(3,1,(i+1))
        ax.append(sub_ax)

    return fig, ax

def get_one_figure_handle():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    return fig, ax

# Show the solver output
def plot_test_result(ax, solver_dict, num_epochs=None):

    assert len(ax) == 3, "Need 3 axis"

    for n in range(len(ax)):
        ax[n].set_xlabel("Epoch")
        if num_epochs is not None:
            ax[n].set_xticks(range(num_epochs))
        if n == 0:
            ax[n].set_title("Training Loss")
        elif n == 1:
            ax[n].set_title("Training Accuracy")
        elif n == 2:
            ax[n].set_title("Validation Accuracy")

    # update data
    for method, solv in solver_dict.items():
        ax[0].plot(solv.loss_history, 'o', label=method)
        ax[1].plot(solv.train_acc_history, '-x', label=method)
        ax[2].plot(solv.val_acc_history, '-x', label=method)

    # Update legend
    for i in range(len(ax)):
        ax[i].legend(loc='upper right', ncol=4)

def plot_3layer_activations(ax, weight_dict):

    assert len(ax) == 3, "Need 3 axis"
    #assert len(weight_dict.keys()) == 3, "Need 3 sets of weights"

    for n in range(len(ax)):
        #grid = vis_weights.vis_grid_img(weight_dict['W' + str(n+1)].transpose(0, 2, 3, 1))
        if n == 0:
            grid = vis_weights.vis_grid_img(weight_dict['W' + str(n+1)].transpose(0, 2, 3, 1))
        elif n >= 1:
            grid = np.zeros_like(weight_dict['W' + str(n+1)])
            #grid = vis_weights.vis_grid_img(weight_dict['W' + str(n+1)].transpose(0, 1))
        ax[n].imshow(grid.astype('uint8'))
        title = "W" + str(n+1)
        ax[n].set_title(title)

# Get the conv layers out of the model
def get_conv_layers(model, verbose=False):
    weight_dict = {}
    for k, v in model.params.items():
        if k[:1] == 'W':
            if len(model.params[k].shape) == 4:
                weight_dict[k] = model.params[k]

    if verbose is True:
        for k, v in weight_dict.items():
            print("%s : %s" % (k, v.shape))

    return weight_dict


def ThreeLayerNet(verbose=True, show_plots=False):
    save_convnet = True
    load_convnet = False
    data_dir = 'datasets/cifar-10-batches-py'
    convnet_path = 'examples/convnet_expr.pkl'

    # Get data
    data = load_data(data_dir, verbose)
    # Set params
    weight_scale = 1e-2
    reg = 1e-3

    # Get a convnet
    # TODO: more flexible convnet
    conv_model = convnet.ThreeLayerConvNet(weight_scale=weight_scale,
                                        hidden_dim=500,
                                        reg=reg)
    if verbose:
        print(conv_model)
    # Get a solver
    conv_solver = solver.Solver(conv_model, data,
                                num_epochs=20,
                                batch_size=50,
                                update_rule='adam',
                                optim_config={'learning_rate': 1e-3},
                                verbose=verbose,
                                checkpoint_dir='examples',
                                checkpoint_name='3ln',
                                print_every=50)
    if load_convnet:        # FIXME : load data.
        print("Loading convnet from file %s" % convnet_path)
        conv_solver.load(convnet_path)

        if verbose is True:
            print("Training %d layer net" % conv_model.num_layers)
        conv_solver.train()

    if save_convnet:
        conv_solver.save(convnet_path)

    # Time to try and visualize what is happening...
    print("break here")
    weight_dict = {'W1': conv_solver.model.params['W1'],
                   'W2': conv_solver.model.params['W2'],
                   'W3': conv_solver.model.params['W3']
                   }
    # Sizes
    print("Layer weight sizes: ")
    for k, v, in weight_dict.items():
        print("%s : %s" % (k, v.shape))
    # Max, min
    print("Layer weight max, min")
    for k, v in weight_dict.items():
        print("%s : max = %f, min = %f" % (k, np.max(v), np.min(v)))

    if show_plots is True:
        fig, ax = get_one_figure_handle()
        grid = vis_weights.vis_grid_img(weight_dict['W1'].transpose(0, 2, 3, 1))
        ax.imshow(grid)
        fig.set_size_inches(5,5)

        # The training loss, accuracy, etc
        tfig, tax = get_figure_handles()
        solver_dict = {'convnet': conv_solver}
        plot_test_result(tax, solver_dict, num_epochs=None)
        plt.show()

    print("done")

def LLayerConv(verbose=True, show_plots=False, solver_filename=None):
    data_dir = 'datasets/cifar-10-batches-py'

    # Get data
    num_train = 1000
    dataset = load_data(data_dir, verbose)
    train_data = {
        'X_train': dataset['X_train'][:num_train],
        'y_train': dataset['y_train'][:num_train],
        'X_val':   dataset['X_val'][:num_train],
        'y_val':   dataset['y_val'][:num_train]
    }
    # Set params
    input_dim = (3, 32, 32)
    weight_scale = 0.06
    learning_rate = 0.07
    #reg = 1e-2
    reg = 0.05
    filter_size = 5
    num_filters = [16, 32, 64]
    hidden_dim = [256, 128]
    num_epochs = 100
    # Get a convnet
    conv_model = convnet.ConvNetLayer(input_dim=input_dim,
                                      hidden_dims=hidden_dim,
                                      num_filters=num_filters,
                                      weight_scale=weight_scale,
                                      reg=reg,
                                      filter_size=filter_size,
                                      use_batchnorm=True,
                                      verbose=verbose)
    if verbose:
        print(conv_model)
    # Get a solver
    conv_solver = solver.Solver(conv_model, train_data,
                                num_epochs=num_epochs,
                                batch_size=10,
                                update_rule='adam',
                                optim_config={'learning_rate': learning_rate},
                                verbose=verbose,
                                print_every=50,
                                checkpoint_name='c2-32-32-f2-256-256',
                                checkpoint_dir='examples')
    if verbose is True:
        print("Training %d layer net" % conv_model.num_layers)
    conv_solver.train()
    conv_solver.save(solver_filename)

    if show_plots is True:
        # The training loss, accuracy, etc
        tfig, tax = get_figure_handles()
        solver_dict = {'convnet': conv_solver}
        plot_test_result(tax, solver_dict, num_epochs=num_epochs)
        plt.show()
    if verbose:
        print(conv_model)
    # Get a solver
    conv_solver = solver.Solver(conv_model, train_data,
                                num_epochs=num_epochs,
                                batch_size=50,
                                update_rule='adam',
                                optim_config={'learning_rate': 1e-3},
                                verbose=verbose,
                                print_every=50)
    conv_solver.train()
    # Plot results
    #fig, ax = get_one_figure_handle()
    #grid = vis_weights.vis_grid_img(weight_dict['W1'].transpose(0, 2, 3, 1))
    #ax.imshow(grid)
    #fig.set_size_inches(5,5)
    # save the data
    solver_file = "examples/conv_solver_%d_epochs.pkl" % num_epochs
    conv_solver.save(solver_file)

    # The training loss, accuracy, etc
    tfig, tax = get_figure_handles()
    solver_dict = {'convnet': conv_solver}
    plot_test_result(tax, solver_dict, num_epochs=num_epochs)
    plt.show()

    print("done")


if __name__ == "__main__":
    fname = "conv2fc2-40-epoch.pkl"
    #ThreeLayerNet()
    LLayerConv(solver_filename=fname)
