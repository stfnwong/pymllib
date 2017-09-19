"""
NN_GUI

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import matplotlib.pyplot as plt
import numpy as np
import neural_net
import linear
# data utils
import neural_net_utils as nnu

# Debug
#from pudb import set_trace; set_trace()


class NNGUI(object):
    def __init__(self):
        # Data params
        self.data_dims = 2
        self.num_classes = 3

        # Classifiers
        self.nn_classifier = None
        self.linear_classifier = None

        # some figures
        self.fig_synth_data = None
        self.fig_nn_surf = None

    def init_linear_classifier(self):

        self.linear_classifier = linear.LinearClassifier()
        self.linear_classifier.init_params(self.data_dims, self.num_classes)

    def gen_synth_data(self, N, D, K, theta):

        circle_data = nnu.create_circle_data(N, D, K, theta)
        spiral_data = nnu.create_spiral_data(N, D, K, theta)

        datasets = {}
        datasets['circle'] = circle_data
        datasets['spiral'] = spiral_data

        return datasets


    def vis_synth_data(self, X, y):

        self.fig_synth_data = plt.figure()
        ax = self.fig_synth_data.add_subplot(1,1,1)
        ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)

        self.fig_synth_data.canvas.draw()
        #plt.show()




if __name__ == "__main__":

    nngui = NNGUI()
    datasets = nngui.gen_synth_data(100, 2, 3, 0.2)
    nngui.vis_synth_data(datasets['spiral'][0], datasets['spiral'][1])
    nngui.vis_synth_data(datasets['circle'][0], datasets['circle'][1])
    plt.show()


