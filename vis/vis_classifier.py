"""
VIS_CLASSIFIER
Visualizations for classifiers.

Stefan Wong 2017
"""

import numpy as np
import matplotlib.pyplot as plt

# Debug
#from pudb import set_trace; set_trace()

def init_classifier_plot(figtitle="", fignum=None):

    if fignum is not None and type(fignum) is int:
        fig = plt.figure(fignum)
    else:
        fig = plt.figure()

    # Create axis. For now
    ax = fig.add_subplot(1,1,1)

    return (fig, ax)


def vis_weights():
    print("TODO: visualize weights")

def vis_classifier_simple(params, data, ax, title_text=None):
    """
    INPUTS:
        params:
            dict of Weights and biases. Params["W"] contains a tuple of
            arrays, each representing a layer in the network
        data:
            dict of data points
    """

    if(ax is None):
        return

    ax.clear()

    # Classifier weights
    W = params['W']
    b = params['b']
    # Dataset
    X = data['X']
    y = data['y']

    # TODO : Set how the weights are mapped.

    h = 0.02
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    if(type(W) is tuple or type(W) is list):            # list hack here for testing purposes
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
    ax.set_title(str(title_text))



if __name__ == "__main__":

    fig, ax = init_classifier_plot()


