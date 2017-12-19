"""
K-NEAREST NEIGHBOURS CLASSIFIER

Stefan Wong 2017
"""

import numpy as np
from collections import Counter

# Debug
#from pudb import set_trace; set_trace()

class KNNClassifier(object):
    def __init__(self):
        self.k = 1

    def train(self, X, y):
        """
        TRAIN

        INPUTS:
        X:
            Array of (N, D) where N is the number of training samples
            and D is the dimension of the data.
        y:
            Array of (,N) containing the training labels where y[i]
            is the label for X[i]
        """
        self.X_train = X
        self.y_train = y

    def compute_dist(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

    def predict(self, X):
        """
        PREDICT

        INPUTS:
        X :
             Array of (N, D) where N is the number of test points
             and D is the dimension of the data
        """
        dists = self.compute_dist(X)
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_train):
            ynn = []        # closest y
            idx_array = np.argsort(dists[i, :], axis=0)
            for n in range(self.k):
                ynn.append(self.y_train[idx_array[n]])

            y_pred[i] = Counter(ynn).most_common(1)[0][0]

        return y_pred

