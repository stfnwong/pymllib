"""
TEST_CAPTIONING_RNN

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from pymllib.utils import error
# Unit under test
from pymllib.layers import rnn_layers
from pymllib.classifiers import captioning_rnn
from pymllib.solver import captioning_solver

# Debug
from pudb import set_trace; set_trace()



class TestCaptioningRNN(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-6
        self.verbose = True

    def test_step_forward(self):
        print("\n======== TestCaptioningRNN.test_step_forward:")

        N = 3
        D = 10
        H = 4
        X = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn_layers.rnn_step_forward(X, prev_h, Wx, Wh, b)
        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [ 0.66854692,  0.79562378,  0.8775553,   0.92795967],
            [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

        err = error.rel_error(expected_next_h, next_h)
        print('Relative error : %f' % err)
        self.assertLessEqual(err, self.eps)

        print("======== TestCaptioningRNN.test_step_forward: <END> ")


if __name__ == '__main__':
    unittest.main()
