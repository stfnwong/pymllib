"""
SOLVER_UTILS
Utils to load solvers and checkpoints from disk individually or in batches

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np

import pymllib.solver.solver as solver
import pymllib.utils.solver_utils as solver_utils




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    cpath = "/home/kreshnik/compucon/machine-learning/models/conv3fc2-2017-11-12-01"

    # Get a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for n in range(num_checkpoints):
        cname = cpath + '/' + "conv3fc2-check_epoch_%d.pkl" % n
        solv = solver.Solver(None, None)
        solv.load(cname)
        solver_utils.plot_model_first_layer(ax, solv.model, cname)
        plt.show()


