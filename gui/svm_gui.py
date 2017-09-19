"""
SVM_GUI
GUI for presenting linear SVM with matplotlib and PyQT

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

#import matplotlib
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
#from matplotlib.figure import Figure
import numpy as np

# CLASSIFIER INTERNALS
import svm
import svm_utils
#from utils import svm_utils


# Debug
#from pudb import set_trace; set_trace()

"""
SVM_GUI

Holds a classifier and the various parts needed to display
its internals in a meaningful way.
"""
class SVMGUI(object):
    def __init__(self, parent=None):
        # Classifier
        self.svm_classifier = svm.LinearSVM()
        self.classes = svm_utils.classes
        self.datasets = {}


        # Figure to show some sample classes
        self.fig_vis_class = []
        #self.ax_vis_class = self.fig_vis_class.add_subplot(1,1,1)

    def plot(self, x, y, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        plot_data, = ax.plot(x, y, **kwargs)
        # TODO : setting the axie labels?

        return plot_data

    def subplot(self, data, fig=None, index=111):

        if fig is None:
            fig = plt.figure()     #  This could be annoying....
        ax = fig.add_subplot(index)
        ax.plot(data)       # TODO : this poses problems for me because I want images....


    def visualise_classes(self, X_train, y_train, classes):

        num_classes = len(classes)
        samples_per_class = 7

        self.fig_vis_class, ax = plt.subplots(nrows=samples_per_class, ncols=num_classes)
        print(ax.shape)
        for y, class_title in enumerate(classes):
            idxs = np.flatnonzero(y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)

            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                ax[i,y].imshow(X_train[idx].astype('uint8'))
                ax[i,y].set_xticks([])
                ax[i,y].set_yticks([])
                #ax[i,y].axis('off')
                #ax[plt_idx].axis('off')

                #plt.subplot(samples_per_class, num_classes, plt_idx)
                #plt.imshow(X_train[idx].astype('uint8'))
                #plt.axis('off')
                if i == 0:
                    ax[i,y].set_title(class_title)
        self.fig_vis_class.suptitle("Visualization of classes in CIFAR10", fontsize=20)
        self.fig_vis_class.canvas.draw()
        plt.show()

    def load_data(self, cifar10_dir):

        self.datasets = svm_utils.gen_datasets(cifar10_dir)

        # Show some data
        self.visualise_classes(self.datasets['X_train'], self.datasets['y_train'], self.classes)



    def draw_test(self):

        # Sample data
        dt = 0.001
        t = np.arange(0.0, 10.0, dt)
        r = np.exp(-t[:1000]/0.05)
        x = np.random.randn(len(t))
        s = np.convolve(x, r)[:len(x)] * dt

        self.fig = plt.figure()
        ax = self.fig.add_subplot(1,1,1)
        ax.set_title("TEST TITLE")
        ax.plot(t, s)
        ax.set_ylabel('some numbers')

        # Bring up the canvas
        self.fig.canvas.draw()
        plt.show()





if __name__ == "__main__":

    cifar10_dir = 'datasets/cifar-10-batches-py'

    gui = SVMGUI()
    gui.load_data(cifar10_dir)
    #gui.draw_test()
