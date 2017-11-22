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


def ex_plot_solver_weights(ax, fname, title=None):
    """
    EX_PLOT_SOLVER_WEIGHTS
    Example showing how to plot the first layer weights
    in a solver object
    """

    if title is None:
        title = "Layer 1 weights"
    solv = solver.Solver(None, None)
    solv.load_checkpoint(fname)
    vis_solver.plot_model_first_layer(ax, solv.model, cname)
    ax.set_title(title)


def ex_plot_sequence(ax, path, fname, num_checkpoints, prefix=None, step=1, pause_time=0.01):
    """
    EX_PLOT_SEQUENCE
    Example wrapper for vis_solver.plot_model_first_layer showing a possible
    inner loop for a weight visualization animation
    """

    if type(num_checkpoints) is tuple:
        if len(num_checkpoints) > 2:
            raise ValueError("Cannot accept more than 2 limits for num_checkpoints")
        if num_checkpoints[0] == 0:
            n_min = 1
        else:
            n_min = int(num_checkpoints[0])
        n_max = int(num_checkpoints[1])
    else:
        n_min = 1
        n_max = int(num_checkpoints)

    # Check input arguments
    if type(path) is not list:
        path = [path]

    if type(fname) is not list:
        fname = [fname]

    # Iterate over all files and generate animations
    for p in path:
        for f in fname:
            for n in range(n_min, n_max, step):
                if prefix is not None:
                    cname = '%s/%s/%s_epoch_%d.pkl' % (prefix, p, f, int(n))
                else:
                    cname = '%s/%s_epoch_%d.pkl' % (p, f, int(n))
                solv = solver.Solver(None, None)
                solv.load_checkpoint(cname)
                title = '%s (epoch %d)' % (f, n)
                vis_solver.plot_model_first_layer(ax, solv.model, title=title)
                plt.pause(pause_time)
                plt.draw()


def ex_vis_solver_compare(ax, path, fname, epoch_num, prefix=None):
    """
    EX_VIS_SOLVER_COMPARE
    Visualize a series of solutions superimposed on a single plot.
    Each solution checpoint is read in turn an plotted on a single
    graph. The legend is created using the __repr__() result for each
    solver object.

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
            solv = solver.Solver(None, None)
            solv.load_checkpoint(cname)
            solver_dict[f] = solv
            #vis_solver.plot_model_first_layer(ax, solv.model, cname)
    vis_solver.plot_solver_compare(ax, solver_dict)


if __name__ == "__main__":

    solv_fig, solv_ax = vis_solver.get_train_fig()
    w_fig, w_ax = vis_solver.get_weight_fig()

    prefix = "/home/kreshnik/Documents/compucon/machine-learning/models"
    cpath = ["conv-net-train-2017-11-15-01", "conv-net-train-2017-11-15-02"]
    cname = ['c16-fc256-fc10-net', 'c16-c32-fc256-fc10-net', 'c16-c32-c64-fc256-fc256-fc10-net', 'c16-c32-c64-c128-fc256-fc256-fc10-net']
    ex_vis_solver_compare(solv_ax, cpath, cname, 100, prefix)
    ex_plot_sequence(w_ax, cpath, cname, (1, 100), prefix=prefix, step=10)
    plt.show()
