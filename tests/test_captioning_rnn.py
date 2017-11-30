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
from pymllib.utils import check_gradient
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

    def test_step_backward(self):
        print("\n======== TestCaptioningRNN.test_step_forward:")

        N = 4
        D = 5
        H = 6
        X = np.random.randn(N, D)
        h = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)
        dnext_h = np.random.randn(*out.shape)

        fx = lambda x:      rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fh = lambda prev_h: rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fWx = lambda Wx:    rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fHw = lambda Wh:    rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]
        fb = lambda b:      rnn_layers.rnn_step_forward(X, h, Wx, Wh, b)[0]

        dx_num = check_gradient.eval_numerical_gradient_array(fx, X, dnext_h)
        dprev_h_num = check_gradient.eval_numerical_gradient_array(fh, X, dnext_h)
        dWx_num = check_gradient.eval_numerical_gradient_array(fWx, X, dnext_h)
        dWh_num = check_gradient.eval_numerical_gradient_array(fWh, X, dnext_h)
        db_num = check_gradient.eval_numerical_gradient_array(fb, X, dnext_h)

        dx, dprev_h, dWx, dWh, db = rnn_layers.rnn_step_backward(dnext_h, cache)

        dx_err = error.rel_error(dx, dx_num)
        dprev_h_err = error.rel_error(dprev_h, dprev_h_num)
        dwx_err = error.rel_error(dWx, dWx_num)
        dwh_err = error.rel_error(dWh, dHw_num)
        db_err = error.rel_error(db, db_num)

        print("dx_err : %f" % dx_err)
        print("dprev_h_err : %f" % dprev_h_err)
        print("dwx_err : %f" % dwx_err)
        print("dhx_err : %f" % dhx_err)
        print("db_err : %f" % db_err)

        self.assertLessEqual(dx_err, self.eps)
        self.assertLessEqual(dprev_h_err, self.eps)
        self.assertLessEqual(dwx_err, self.eps)
        self.assertLessEqual(dwh_err, self.eps)
        self.assertLessEqual(db_err, self.eps)

        print("======== TestCaptioningRNN.test_step_forward: <END> ")


if __name__ == '__main__':
    unittest.main()
