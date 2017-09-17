"""
CHECK_GRADIENT
Perform a numerical gradient check. The format for
this is mainly taken from CS231n . Written for Python3

Stefan Wong 2017
"""

import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose=False, h=1e-5):
    """
    A naive implementation of numerical gradient of f at x

    INPUTS:
        f -
            A function that takes a single argument
        x -
            Point (numpy array) to evaluate gradient at
    OUTPUTS:
        grad -
            An array of the same shape as x containing the
            computed gradient
    """

    grad = np.zeros_like(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # eval the function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h          # increment by h
        fxph = f(x)                 # eval f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)                 # eval f(x - h)
        x[ix] = oldval              # restore

        # Compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()       # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    An array-based  implementation of numerical gradient of f at x.
    Evaluates a numeric gradient for a function that accepts a
    numpy array and returns a numpy array

    INPUTS:
        f - A function that takes a single argument
        x - Point (numpy array) to evaluate gradient at
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()       # f(x + h)
        x[ix] = oldval - h
        neg = f(x).copy()       # f(x - h)
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()

    return grad


def grad_compare_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    GRAD_COMPARE_SPARSE

    Sample and compare some random elements
    """

    err = np.zeros((1, num_checks))

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        err[i] = rel_error

    return err
