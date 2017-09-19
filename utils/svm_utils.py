"""
SVM UTILS
Utilities for an SVM demo.

Stefan Wong 2017
"""

import numpy as np
from utils.data_utils import load_CIFAR10

# Debug
#from pudb import set_trace; set_trace()


# TODO : This should go into the GUI
#def visualise_classes(X_train, classes):
#
#    num_classes = len(classes)
#    samples_per_class = 7
#
#    for y, cls in enumerate(classes):
#        idxs = np.flatnonzero(y_train == y)
#        idxs = np.random.choice(idxs, samples_per_class, replace=False)
#        for i, idx in enumerate(idxs):
#            plt_idx = i * num_classes + y + 1
#            plt.subplot(samples_per_class, num_classes, plt_idx)
#            plt.imshow(X_train[idx].astype('uint8'))
#            plt.axis('off')
#            if i == 0:
#                plt.title(cls)


# Selection of classes from CIFAR10 dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def gen_datasets(cifar10_dir):

    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # NOTE : Taken directly from the python notebook
    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

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
    datasets['X_train' : X_train]
    datasets['X_val' : X_val]
    datasets['X_dev' : X_dev]
    datasets['X_test' : X_test]
    datasets['y_dev' : y_dev]
    datasets['y_test' : y_test]
    datasets['y_val' : y_val]

    return datasets
