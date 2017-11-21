"""
VIS_SOLVER
Visualize a solution parameters.

Stefan Wong 2017
"""


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import pymllib.vis.vis_weights as vis_weights

def get_train_fig():
    fig = plt.figure()
    ax = []
    for i in range(3):
        subax = fig.add_subplot(3, 1, (i+1))
        ax.append(subax)

    return fig, ax

def get_weight_fig():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax

def plot_solver_compare(ax, solver_dict, num_epochs=None):
    """
    Make a comparative plot for any number of solvers. Solver
    objects are stored in a dictionary, where the key name is
    uesd in the legend to record the solution method
    """

    assert type(solver_dict) is dict, "Solver must be in a dictionary"
    assert len(ax) == 3, "Need 3 axis"

    for n in range(len(ax)):
        if n == 0:
            ax[n].set_xlabel("Iterations")
            ax[n].set_ylabel("Loss")
        else:
            ax[n].set_xlabel("Epoch")
            ax[n].set_ylabel("Accuracy")
        #if num_epochs is not None:
        #    ax[n].set_xticks(range(num_epochs))
        if n == 0:
            ax[n].set_title("Training Loss")
        elif n == 1:
            ax[n].set_title("Training Accuracy")
        elif n == 2:
            ax[n].set_title("Validation Accuracy")

    # update data
    for slabel, solv in solver_dict.items():
        ax[0].plot(solv.loss_history, 'o', label=slabel, markersize=2)
        ax[1].plot(solv.train_acc_history, label=slabel)
        ax[2].plot(solv.val_acc_history, label=slabel)

    # Update legend
    for i in range(len(ax)):
        if i == 0:
            ax[i].legend(loc='upper right', ncol=4)
        else:
            ax[i].legend(loc='lower right', ncol=4)


def plot_solver(ax, solv, num_epochs=None, method=None):
    """
    Make a single plot for the output of one solver
    """

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
    ax[0].plot(solv.loss_history, 'o', label=method)
    ax[1].plot(solv.train_acc_history, '-x', label=method)
    ax[2].plot(solv.val_acc_history, '-x', label=method)

    # Update legend
    for i in range(len(ax)):
        ax[i].legend(loc='upper right', ncol=4)

def plot_model_first_layer(ax, model, fname):

    for k, v in model.params.items():
        # We just do first layer for now
        if k[:2] == 'W1':
            w1 = v

    grid = vis_weights.vis_grid_img(w1.transpose(0, 2, 3, 1))
    ax.imshow(grid)
    ax.axis('off')
