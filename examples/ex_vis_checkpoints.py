import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np
import matplotlib.pyplot as plt

from pymllib.solver import solver
from pymllib.vis import vis_solver


def plot_single(cname):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    title = "Layer 1 weights"
    solv = solver.Solver(None, None)
    solv.load_checkpoint(cname)
    vis_solver.plot_model_first_layer(ax, solv.model, cname)
    ax.set_title(title)

    # Also plot the training results
    fig2 = plt.figure()
    ax2 = [ ]
    for i in range(3):
        subax = fig2.add_subplot(3, 1, (i+1))
        ax2.append(subax)
    vis_solver.plot_solver(ax2, solv)
    plt.show()



def vis_weight_sequence(ax, path, fname, num_checkpoints, prefix=None):        # TODO ; add params
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
        num_checkpoints:
            The number of *.pkl files to read. This may a scalar, in which case the
            method reads files from  <fname>_epoch_1.pkl to <fname>_epoch_n.pkl where
            n = num_checkpoints. Alternatively, this argument may be a tuple of (m, n)
            in which case the method reads files from <fname>_epoch_m.pkl through
            <fname>_epoch_n.pkl.
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


    # Iterate over all files and generate animations
    for p in path:
        for f in fname:
            for n in range(n_min, n_max):
                title = "Layer 1 Weights (epoch %d)" % (n+1)
                epoch_str = '_epoch_%d.pkl' % (n)
                if prefix is not None:
                    fname = str(prefix) + '/' + str(p) + '/' + str(f) + str(epoch_str)
                else:
                    fname = str(p) + '/' + str(f) + str(epoch_str)
                solv = load_solver(fname)
                vis_solver.plot_model_first_layer(ax, solv.model, cname)
                ax.set_title(title)
                plt.pause(1)
                plt.draw()

    # At the final checkpoint, plot the overall training results
    # TODO : Move this to another routine one layer up
    #fig2 = plt.figure()
    #ax2 = [ ]
    #for i in range(3):
    #    subax = fig2.add_subplot(3, 1, (i+1))
    #    ax2.append(subax)
    #vis_solver.plot_solver(ax2, solv)
    #plt.show()





if __name__ == "__main__":
    prefix = "/home/kreshnik/Documents/compucon/machine-learning/models"
    cpath = ["conv-net-train-2017-11-15-01", "conv-net-train-2017-11-15-02"]
    cname = ['c16-fc256-fc10-net', 'c16-c32-fc256-fc10-net', 'c16-c32-c64-fc256-fc256-fc10-net', 'c16-c32-c64-c128-fc256-fc256-fc10-net']


    solver_dict = {}
    for p in cpath:
        for n in cname:
            cpoint_filename = prefix + '/' + p + '/' + n + '_epoch_100.pkl'
            solv = load_solver(cpoint_filename)
            solver_dict[n + p[-1]] = solv

    # Plot solvers against each other
    fig, ax = vis_solver.get_train_fig()
    vis_solver.plot_solver_compare(ax, solver_dict)
    fig.tight_layout()
    plt.show()

    #plot_sequence()
    #plot_single(cpoint_filename)
