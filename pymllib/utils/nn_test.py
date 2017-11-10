"""
Train a 2-layer network on spiral data

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../vis')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import numpy as np
import matplotlib.pyplot as plt
# Internal libs
import neural_net_utils as nnu
import fcnet
import twolayer_modular
import vis_classifier as vis

#Debug
from pudb import set_trace; set_trace()



def train_to_end(net, X, y, ax, num_iter=10000):

    loss_param = net.train(X, y, num_iter=num_iter, cache_loss=True)
    # TODO : Show the loss parameters
    params = {'W': [net.params['W1'], net.params['W2']],
              'b': [net.params['b1'], net.params['b2']]}
    data = {'X': X,
            'y': y}
    vis.vis_classifier_simple(params, data, ax, title_text="Classifer")
    plt.show()


def train_with_update(net, X, y, ax, num_iter, update_every=1000, cache_loss=False):
    """
    Puts the training loop here so that we can incrementally plot
    the decision boundary
    """

    if cache_loss is True:
        loss_cache = np.zeros(num_iter)

    for n in range(num_iter):
        loss, grads = net.loss(X, y)
        net.param_update(grads)

        if cache_loss is True:
            loss_cache[n] = loss

        # update the plots
        if n % update_every == 0:
            params = {'W': [net.params['W1'], net.params['W2']],
                    'b': [net.params['b1'], net.params['b2']]}
            data = {'X': X,
                    'y': y}
            title = "Decision boundary (Iteration %d)" % (n+1)
            vis.vis_classifier_simple(params, data, ax, title_text=title)
            plt.draw()
            plt.pause(0.01)
            # TODO : Show loss over time

    # Show final boundaries
    params = {'W': [net.params['W1'], net.params['W2']],
            'b': [net.params['b1'], net.params['b2']]}
    data = {'X': X,
            'y': y}
    title = "Final Decision boundary (%d Iterations)" % (n+1)
    vis.vis_classifier_simple(params, data, ax, title_text=title)
    plt.show()


# ======== ENTRY POINT ======== #
if __name__ == "__main__":

    animate = True
    # Generate data
    N = 200         # Number of data points
    h = 100          # Size of hidden dimension
    D = 2           # Dimension of data
    K = 3           # Number of classes
    theta = 0.35
    spiral_data = nnu.create_spiral_data(N, D, K, theta)
    X = spiral_data[0]
    y = spiral_data[1]

    # Get plot handles
    fig, ax = vis.init_classifier_plot()

    std = 1e-2
    training_iter = int(2e4)
    step_size = 6e-2
    #  Get a network
    net = twolayer_modular.TwoLayerNet(input_dim=D, hidden_dim=h,
                num_classes=K, weight_scale=std, step_size=step_size,
                verbose=True)

    # Show the plots
    if animate is True:
        train_with_update(net, X, y, ax, training_iter, update_every=1000)
    else:
        train_to_end(net, X, y, ax, training_iter)

