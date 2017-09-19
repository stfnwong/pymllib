"""
SUPPORT VECTOR MACHINE
This is my re-implementation of an SVM for my own personal
edification. This software comes with no guarantee of
optimality or performance

Stefan Wong 20177
"""

import numpy as np

# Debug
#from pudb import set_trace; set_trace()

class LinearSVM(object):
    ## TODO ; I don't know at this stage whether its
    # worth encoding all the hyperparameters here, and
    # in fact I don't even know if the SVM is "complex"
    # enough to need much encapsulation anyway
    def __init__(self):
        self.delta = 1

    def compute_loss(self, W, X, y, reg):

        # Compute the loss and loss gradient
        dW = np.zeros(W.shape)
        loss = 0.0
        num_train = X.shape[0]

        scores = np.dot(X, W)
        yi_scores = scores[np.arange(scores.shape[0]), y]   #
        margins = np.maximum(0, scores - np.matrix(yi_scores.T + 1))



        return loss, dW


