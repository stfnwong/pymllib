"""
SVM UTILS
Utilities for an SVM demo.

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

import numpy as np
from data_utils import load_CIFAR10
#from utils.data_utils import load_CIFAR10

# Debug
#from pudb import set_trace; set_trace()

# Selection of classes from CIFAR10 dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def gen_datasets(cifar10_dir, num_training=4900, num_validation=1000, num_test=1000, num_dev=500):

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.

    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    datasets = {}
    datasets['X_train'] = X_train
    datasets['X_val'] = X_val
    datasets['X_dev'] = X_dev
    datasets['X_test'] = X_test
    datasets['y_train'] = y_train
    datasets['y_dev'] = y_dev
    datasets['y_test'] = y_test
    datasets['y_val'] = y_val

    return datasets
