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



def plot_sequence():        # TODO ; add params

    #cpath = "/home/kreshnik/Documents/compucon/machine-learning/models/conv3fc2-2017-11-12-01"
    #cpath = "/home/kreshnik/Documents/compucon/machine-learning/models/conv4fc2-2017-11-11-02"

    # Get a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    num_checkpoints = 100        # max = 102

    for n in range(num_checkpoints):
        title = "Layer 1 Weights (epoch %d)" % (n+1)
        #cname = cpath + '/' + "conv3fc2-check_epoch_%d.pkl" % (n+1)
        cname = cpath + '/' + "c16-fc256-fc10-net_epoch_%d.pkl" % (n+1)
        solv = solver.Solver(None, None)
        solv.load_checkpoint(cname)
        vis_solver.plot_model_first_layer(ax, solv.model, cname)
        ax.set_title(title)
        plt.pause(1)
        plt.draw()

    # At the final checkpoint, plot the overall training results
    fig2 = plt.figure()
    ax2 = [ ]
    for i in range(3):
        subax = fig2.add_subplot(3, 1, (i+1))
        ax2.append(subax)
    vis_solver.plot_solver(ax2, solv)
    plt.show()


def load_solver(fname):
    solv = solver.Solver(None, None)
    solv.load_checkpoint(fname)

    return solv


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
