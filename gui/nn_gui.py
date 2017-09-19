"""
NN_GUI

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import time
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
        self.iter_sleep = 1
        self.data_dims = 2
        self.num_classes = 3

        # Classifiers
        self.nn_classifier = None
        self.linear_classifier = None

        # some figures
        self.fig_synth_data = None
        self.fig_classifier = None

    def init_linear_classifier(self):

        self.linear_classifier = linear.LinearClassifier()
        self.linear_classifier.init_params(self.data_dims, self.num_classes)

    def run_linear_classifier(self, X, y):

        self.init_linear_classifier()

        num_iters = 50
        for n in range(num_iters):
            loss, dscores = self.linear_classifier.forward_iter(self.linear_classifier.W, X, y)
            Wout, bout = self.linear_classifier.backward_iter(self.linear_classifier.W, X, dscores)
            self.linear_classifier.b = bout
            self.linear_classifier.W = Wout

            # Plot classifier

            time.sleep(self.iter_sleep)

        # Plot once at end for test purposes
        self.vis_classifier(self.linear_classifier.W, X, y, self.linear_classifier.b)


    def vis_synth_data(self, X, y):

        self.fig_synth_data = plt.figure()
        ax = self.fig_synth_data.add_subplot(1,1,1)
        ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)

        self.fig_synth_data.canvas.draw()
        #plt.show()

    def vis_classifier(self, W, X, y, b):

        h = 0.02
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        self.fig_classifier = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())




if __name__ == "__main__":

    nngui = NNGUI()
    N = 100
    D = 2
    K = 3
    theta = 0.2
    circle_data = nnu.create_circle_data(N, D, K-1, theta)
    spiral_data = nnu.create_spiral_data(N, D, K, theta)
    #nngui.vis_synth_data(circle_data[0], circle_data[1])
    #nngui.vis_synth_data(spiral_data[0], spiral_data[1])

    nngui.run_linear_classifier(spiral_data[0], spiral_data[1])

    plt.show()


