import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np
import matplotlib.pyplot as plt

from pymllib.solver import solver
from pymllib.vis import vis_solver
from pymllib.utils import solver_utils

# Debug
#from pudb import set_trace; set_trace()


def plot_solver_weights(ax, fname, title=None):

    if title is None:
        title = "Layer 1 weights"
    solv = solver.Solver(None, None)
    solv.load_checkpoint(fname)
    vis_solver.plot_model_first_layer(ax, solv.model, cname)
    ax.set_title(title)


# TODO : Visualize an animation of the weights from a set of solvers
def plot_sequence(ax, path, fname, num_checkpoints, prefix=None):        # TODO ; add params

    if type(num_checkpoints) is tuple:
        if len(num_checkpoints) > 2:
            raise ValueError("Cannot accept more than 2 limits for num_checkpoints")
        if num_checkpoints[0] == 0:
            n_min = 1
        else:
            n_min = num_checkpoints[0]
        n_max = num_checkpoints[1]
    else:
        n_min = 1
        n_max = num_checkpoints

    for n in range(n_min, n_max):
        if prefix is not None:
            cname = '%s/%s/%s_epoch_%d.pkl' % (prefix, path, fname, int(n))
        else:
            cname = '%s/%s_epoch_%d.pkl' % (path, fname, int(n))
        solv = solver.Solver(None, None)
        solv.load_checkpoint(cname)
        vis_solver.plot_model_first_layer(ax, solv.model, cname)


def vis_solver_compare(ax, path, fname, epoch_num, prefix=None):
    """
    VIS_WEIGHT_SEQUENCE
    Plot a series of weights as an animated sequence from
    a collection of saved solver states. It is the responsibility
    of the caller to ensure that files exist at the specified paths.

    Inputs
        ax:
            A matplotlib axes onto which to draw the visualization
        path:
            Directory containing solver files. This may be a list of multiple
            directories, in which case the method iterates over each of them in turn.
        fname:
            The name of a given solver file, without the '_epoch_%d.pkl' suffix
        epoch_num:
            Which epoch to load.
        prefix:
            A prefix that is prepended to the filename. This allows, for example, a
            group of subfolders to be traversed that all have the same root.
            Default = None
    """

    # Helper function for loading solver objects
    def load_solver(fname):
        solv = solver.Solver(None, None)
        solv.load_checkpoint(fname)
        return solv

    # Check input arguments
    if type(path) is not list:
        path = [path]

    if type(fname) is not list:
        fname = [fname]

    # Iterate over all files and generate animations
    solver_dict = {}
    for p in path:
        for f in fname:
            epoch_str = '_epoch_%d.pkl' % epoch_num
            if prefix is not None:
                cname = str(prefix) + '/' + str(p) + '/' + str(f) + str(epoch_str)
            else:
                cname = str(p) + '/' + str(f) + str(epoch_str)
            #solv = load_solver(fname)
            print(fname)
            solv = solver.Solver(None, None)
            solv.load_checkpoint(cname)
            solver_dict[f] = solv
            #vis_solver.plot_model_first_layer(ax, solv.model, cname)
    vis_solver.plot_solver_compare(ax, solver_dict)


def update_checkpoints(path, fname, num_epochs=100, prefix=None, verbose=False):
    # Check input arguments
    if type(path) is not list:
        path = [path]

    if type(fname) is not list:
        fname = [fname]

    # Iterate over all files and generate animations
    for p in path:
        for f in fname:
            for n in range(num_epochs):
                epoch_str = '_epoch_%d.pkl' % n
                if prefix is not None:
                    cname = str(prefix) + '/' + str(p) + '/' + str(f) + str(epoch_str)
                else:
                    cname = str(p) + '/' + str(f) + str(epoch_str)
                if verbose is True:
                    print('Converting file %s' % cname)
                #solv = load_solver(fname)
                solv = solver_utils.convert_checkpoint(cname, verbose=verbose)
                solv.save(cname)
                #solver_utils.examine_checkpoint(cname)

if __name__ == "__main__":

    fig, ax = vis_solver.get_train_fig()

    prefix = "/home/kreshnik/Documents/compucon/machine-learning/models"
    cpath = ["conv-net-train-2017-11-15-01", "conv-net-train-2017-11-15-02"]
    cname = ['c16-fc256-fc10-net', 'c16-c32-fc256-fc10-net', 'c16-c32-c64-fc256-fc256-fc10-net', 'c16-c32-c64-c128-fc256-fc256-fc10-net']
    update_checkpoints(cpath, cname, 100, prefix, verbose=True)
    vis_solver_compare(ax, cpath, cname, 100, prefix)
    plt.show()
