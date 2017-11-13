import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import numpy as np

import pymllib.solver.solver as solver
import pymllib.vis.vis_solver as solver_utils
#import pymllib.utils.solver_utils as solver_utils


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    cpath = "/home/kreshnik/Documents/compucon/machine-learning/models/conv3fc2-2017-11-12-01"
    #cpath = "/home/kreshnik/Documents/compucon/machine-learning/models/conv4fc2-2017-11-11-02"

    # Get a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    num_checkpoints = 20        # max = 102

    for n in range(num_checkpoints):
        title = "Layer 1 Weights (epoch %d)" % (n+1)
        cname = cpath + '/' + "conv3fc2-check_epoch_%d.pkl" % (n+1)
        #cname = cpath + '/' + "conv4fc2-check_epoch_%d.pkl" % (n+1)
        solv = solver.Solver(None, None)
        solv.load_checkpoint(cname)
        solver_utils.plot_model_first_layer(ax, solv.model, cname)
        ax.set_title(title)
        plt.pause(1)
        plt.draw()

    # At the final checkpoint, plot the overall training results
    fig2 = plt.figure()
    ax2 = [ ]
    for i in range(3):
        subax = fig2.add_subplot(3, 1, (i+1))
        ax2.append(subax)
    solver_utils.plot_solver(ax2, solv)
    plt.show()
