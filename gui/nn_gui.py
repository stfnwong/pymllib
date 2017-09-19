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
import matplotlib.animation as animation
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

        self.anim_fig = None
        self.anim_ax = None

        self.ax_classifier = None
        self.contour_ax = None
        self.scatter_ax = None

    def init_linear_classifier(self):

        self.linear_classifier = linear.LinearClassifier()
        self.linear_classifier.init_params(self.data_dims, self.num_classes)


    def run_linear_classifier(self, X, y):
        self.anim_fig = plt.figure()
        self.anim_ax = self.anim_fig.add_subplot(111)
        self.init_linear_classifier()
        fig_title = self.anim_ax.set_title("")
        num_iter = 100

        def anim_init():
            h = 0.02
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            Z = np.dot(np.c_[xx.ravel(), yy.ravel()], self.linear_classifier.W) + self.linear_classifier.b
            Z = np.argmax(Z, axis=1)
            Z = Z.reshape(xx.shape)
            self.contour_data, = self.anim_ax.contourf(xx, yy, Z, alpha=0.8)
            self.scatter_data, = self.anim_ax.scatter(X[:,0], X[:,1], animated=True)

            #return self.contour_data, self.scatter_data
            return (self.scatter_data, self.contour_data)

        def anim_frame(i):

            loss, dscores = self.linear_classifier.forward_iter(self.linear_classifier.W, X, y)
            Wout, bout = self.linear_classifier.backward_iter(self.linear_classifier.W, X, dscores)
            self.linear_classifier.b = bout
            self.linear_classifier.W = Wout

            # Set data
            h = 0.02
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            Z = np.dot(np.c_[xx.ravel(), yy.ravel()], self.linear_classifier.W) + self.linear_classifier.b
            Z = np.argmax(Z, axis=1)
            Z = Z.reshape(xx.shape)
            self.contour_data.set_data(xx, yy, Z, alpha=0.8)
            self.scatter_data.set_array(X[:,0], X[:,1])
            fig_title.set_text("Iteration %d" % int(i))

            return (self.scatter_data, self.contour_data)

        ani = animation.FuncAnimation(self.anim_fig, anim_frame, frames=num_iter, interval=10, init_func=anim_init)
        plt.show()

        #num_iters = 50
        #for n in range(num_iters):
        #    loss, dscores = self.linear_classifier.forward_iter(self.linear_classifier.W, X, y)
        #    Wout, bout = self.linear_classifier.backward_iter(self.linear_classifier.W, X, dscores)
        #    self.linear_classifier.b = bout
        #    self.linear_classifier.W = Wout

        #    # Plot classifier

        #    self.vis_classifier(self.linear_classifier.W, X, y, self.linear_classifier.b, n+1)
        #    plt.draw()
        #    time.sleep(self.iter_sleep)

        # Plot once at end for test purposes
        #self.vis_classifier(self.linear_classifier.W, X, y, self.linear_classifier.b)

    def vis_classifier(self, W, X, y, b, ax, it=None):

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

        if(it is not None):
            plt.title('Decision boundary at iteration %d' % it)
        self.fig_classifier.canvas.draw()

    #def vis_synth_data(self, X, y):

    #    self.fig_synth_data = plt.figure()
    #    ax = self.fig_synth_data.add_subplot(1,1,1)
    #    ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)

    #    self.fig_synth_data.canvas.draw()
    #    #plt.show()

    #def init_vis_classifier(self, ax, X, y):

    #    h = 0.02
    #    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    #    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    #    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                         np.arange(y_min, y_max, h))
    #    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
    #    Z = np.argmax(Z, axis=1)
    #    Z = Z.reshape(xx.shape)

    #    self.fig_classifier = plt.figure()
    #    self.ax_classifier = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
    #    self.anim_data = plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    #def init_vis_anim(self, ax):

    #    self.anim_data.set_data([], [], [])
    #    return self.anim_data,

    #def vis_anim(self, i):

    #    self.linear_classifier.forward_iter(self.linear_classifier.W,







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


