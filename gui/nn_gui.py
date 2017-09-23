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
from pudb import set_trace; set_trace()

class NNGUI(object):
    def __init__(self):
        # Data params
        self.num_iter = 500
        self.anim_pause = 0.001
        self.data_dims = 2
        self.num_classes = 4

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

    def init_nn_classifer(self):

        self.nn_classifier = neural_net.NeuralNetwork(h=100, reg=0.021, ss=0.21)
        self.nn_classifier.init_params(self.data_dims, self.num_classes)


    def run_linear_classifier(self, X, y):
        self.anim_fig = plt.figure()
        self.anim_fig.clf()
        self.anim_ax = self.anim_fig.add_subplot(111)
        self.init_linear_classifier()
        fig_title = self.anim_ax.set_title("Linear Classifier")

        for n in range(self.num_iter):
            loss, dscores = self.linear_classifier.forward_iter(X, y)
            Wout, bout = self.linear_classifier.backward_iter(X)
            print("Iter %d, loss = %f" % (n, loss))
            self.vis_classifier(self.linear_classifier.W, X, y, self.linear_classifier.b, self.anim_ax, it=n)
            plt.pause(self.anim_pause)

            #if(n >= self.num_iter - 1):
            #    self.vis_classifier(self.linear_classifier.W, X, y, self.linear_classifier.b, self.anim_ax, it=n)
            #    plt.draw()
            #    #plt.pause(self.anim_pause)

    def run_nn_classifier(self, X, y):
        self.anim_fig = plt.figure()
        self.anim_ax = self.anim_fig.add_subplot(111)
        fig_title = self.anim_ax.set_title("2-Layer Neural Network Classifier")

        self.init_nn_classifer()
        self.nn_classifier.reg = 0.7
        self.nn_classifier.step_size = 0.05
        self.nn_classifier.h_layer_size = 100
        # Save predictions
        y_pred = []

        for n in range(self.num_iter):

            loss, dscores = self.nn_classifier.forward_iter(X, y)
            dW, db = self.nn_classifier.backward_iter(X)
            #grads = {}
            #grads['dW'] = dW
            #grads['db'] = db
            #self.nn_classifier.train_iter(X, y, loss, grads)
            y_pred.append(self.nn_classifier.predict_iter(X))
            print("Iter %d, loss = %f" % (n, loss))

            VisW = (self.nn_classifier.W1, self.nn_classifier.W2)
            Visb = (self.nn_classifier.b1, self.nn_classifier.b2)
            self.vis_classifier(VisW, X, y, Visb, self.anim_ax, it=n)
            plt.draw()
            plt.pause(self.anim_pause)

            #if(n >= self.num_iter - 1):
            #    VisW = (self.nn_classifier.W1, self.nn_classifier.W2)
            #    Visb = (self.nn_classifier.b1, self.nn_classifier.b2)
            #    self.vis_classifier(VisW, X, y, Visb, self.anim_ax, it=n)
            #    plt.show()
            #    plt.draw()
            #    plt.pause(self.anim_pause)


    def tune_nn_params(self, X, y):

        hsizes = [10, 20, 50, 100, 150, 200, 250]
        lrates = [1e-3, 1e-2, 1e-1]
        rstrs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        stat_history = []

        for h in hsizes:
            for l in lrates:
                for r in rstrs:
                    self.init_nn_classifer()
                    self.nn_classifier.h_layer_size = h
                    self.nn_classifier.step_size = l
                    self.nn_classifier.reg = r

                    print("Hyperparams")
                    print("Hsize : %f" % (h))
                    print("Lrate : %f" % (l))
                    print("Reg   : %f" % (r))

                    for i in range(self.num_iter):
                        loss, dscores = self.nn_classifier.forward_iter(X,y)
                        dW, db = self.nn_classifier.backward_iter(X)
                        y_pred = self.nn_classifier.predict_iter(X)

                    train_acc = np.mean(y_pred == y)
                    stats = {}
                    stats['y_pred'] = y_pred
                    stats['hsize'] = h
                    stats['lrate'] = l
                    stats['reg'] = r
                    stats['loss'] = loss
                    stats['train_acc'] = train_acc
                    stat_history.append(stats)

        # Plot some stats
        fig = plt.figure(1)
        acc_yaxis = np.zeros(len(stat_history))
        for n in range(len(stat_history)):
            acc_yaxis[n] = stat_history[n]['train_acc']
        #plt.plot(acc_yaxis)
        #plt.plot(np.arange(1, len(stat_history)), acc_yaxis)
        #plt.show()

        # Find the 'best' accuracy result
        best_acc_idx = np.argmax(acc_yaxis, axis=0)

        print(stat_history[best_acc_idx])
        return stat_history[best_acc_idx]



    def vis_classifier(self, W, X, y, b, ax, it=None):

        if(ax is None):
            return

        ax.clear()

        h = 0.02
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        if(type(W) is tuple):
            Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W[0]) + b[0]), W[1]) + b[1]
        else:
            Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        #self.fig_classifier = plt.figure()
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

        if(it is not None):
            ax.set_title('Decision boundary at iteration %d' % it)
        #plt.draw()
        #self.anim_fig.canvas.draw()

    def vis_synth_data(self, X, y, ax=None):

        if ax is None:
            return

        ax.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
        ax.set_title('Data viz')
        plt.draw()


if __name__ == "__main__":

    nngui = NNGUI()
    N = 100             # Number of points per class
    D = 2               # Dimension of data
    K = 3               # Number of classes
    nngui.num_classes = K
    nngui.data_dims = D
    theta = 0.3
    circle_data = nnu.create_circle_data(N, D, K-1, theta)
    spiral_data = nnu.create_spiral_data(N, D, K, theta)
    # Sanity check data
    print(spiral_data[0].shape)
    print(spiral_data[1].shape)

    #vis_fig = plt.figure()
    #vis_ax = vis_fig.add_subplot(111)
    #nngui.vis_synth_data(circle_data[0], circle_data[1])
    #nngui.vis_synth_data(spiral_data[0], spiral_data[1], ax=vis_ax)
    #plt.pause(1)
    nngui.num_iter = 500
    nngui.tune_nn_params(spiral_data[0], spiral_data[1])
    #nngui.run_linear_classifier(spiral_data[0], spiral_data[1])
    nngui.run_nn_classifier(spiral_data[0], spiral_data[1])

