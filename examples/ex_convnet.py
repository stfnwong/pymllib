"""
EX_CONVNET
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


def main():
    verbose = True
    save_convnet = False
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
                                num_epochs=1,
                                batch_size=50,
                                update_rule='adam',
                                optim_config={'learning_rate': 1e-3},
                                verbose=verbose,
                                print_every=50)
    # TODO : Maybe it should be possible to have an empty solver into which we
    # load data....
    if load_convnet:
        print("Loading convnet from file %s" % convnet_path)
        conv_solver.load(convnet_path)

    # TODL string formatting is not good here...
    #if verbose:
    #    print(conv_solver)
    # Ready to train
    if load_convnet is False:
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

    fig, ax = get_one_figure_handle()
    grid = vis_weights.vis_grid_img(weight_dict['W1'].transpose(0, 2, 3, 1))
    ax.imshow(grid.astype('uint8'))
    #fig.axis('off')
    fig.set_size_inches(5,5)
    plt.show()
    #fig, ax = get_figure_handles()
    #plot_3layer_activations(ax, weight_dict)
    #plt.show()

    print("done")

if __name__ == "__main__":
    main()