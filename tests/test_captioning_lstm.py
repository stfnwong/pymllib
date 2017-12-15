"""
TEST_CAPTIONING_LSTM

Stefan Wong 2017
"""



import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pymllib.utils import error
from pymllib.utils import check_gradient
from pymllib.utils import rnn_utils
from pymllib.utils import coco_utils
from pymllib.utils import image_utils
# Unit under test
from pymllib.layers import rnn_layers
from pymllib.classifiers import captioning_rnn
from pymllib.solver import captioning_solver

# Debug
#from pudb import set_trace; set_trace()


def load_data(verbose=False):
    data = coco_utils.load_coco_data(pca_features=True)
    if verbose:
        for k, v in data.items():
            if type(v) == np.ndarray:
                print('%s : %s, %s, %%d' % (k, type(v), v.shape, v.dtype))
            else:
                print('%s : %s (%s)' % (k, type(v), len(v)))

    return data


class TestCaptioningLSTM(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-6
        self.dtype = np.float64
        self.verbose = True
        self.draw_figures = True

    def test_lstm_step_forward(self):
        print("\n======== TestCaptioningLSTM.test_lstm_step_forward:")

        N = 3
        D = 4
        H = 5
        X = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)
        prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)
        prev_c = np.linspace(-0.4, 0.9, num=N*H).reshape(N, H)
        Wx = np.linspace(-2.1, 1.3, num=4*D*H).reshape(D, 4 * H)
        Wh = np.linspace(-0.7, 2.2, num=4*H*H).reshape(H, 4 * H)
        b = np.linspace(0.3, 0.7, num=4*H)

        next_h, next_c, cache = rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)

        expected_next_h = np.asarray([
                [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],
                [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],
                [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]])
        expected_next_c = np.asarray([
                [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],
                [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],
                [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]])
        h_err = error.rel_error(next_h, expected_next_h)
        c_err = error.rel_error(next_c, expected_next_c)

        self.assertLessEqual(h_err, self.eps)
        self.assertLessEqual(c_err, self.eps)
        print('h_err : %f' % h_err)
        print('c_err : %f' % c_err)

        print("\n======== TestCaptioningLSTM.test_lstm_step_forward: <END>")

    def test_lstm_step_backward(self):
        print("\n======== TestCaptioningLSTM.test_lstm_step_backward:")

        N = 4
        D = 5
        H = 6
        X = np.random.randn(N, D)
        prev_h = np.random.randn(N, H)
        prev_c = np.random.randn(N, H)
        Wx = np.random.randn(D, 4 * H)
        Wh = np.random.randn(H, 4 * H)
        b = np.random.randn(4 * H)

        next_h, next_c, cache = rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)
        dnext_h = np.random.randn(*next_h.shape)
        dnext_c = np.random.randn(*next_c.shape)

        fx_h  = lambda  x: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]
        fh_h  = lambda  h: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]
        fc_h  = lambda  c: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]
        fWx_h = lambda Wx: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]
        fWh_h = lambda Wh: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]
        fb_h  = lambda  b: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[0]

        fx_c  = lambda  x: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]
        fh_c  = lambda  h: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]
        fc_c  = lambda  c: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]
        fWx_c = lambda Wx: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]
        fWh_c = lambda Wh: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]
        fb_c  = lambda  b: rnn_layers.lstm_step_forward(X, prev_h, prev_c, Wx, Wh, b)[1]

        # Evaluate gradients
        num_grad = check_gradient.eval_numerical_gradient_array
        dx_num = num_grad(fx_h, X, dnext_h) + num_grad(fx_c, X, dnext_c)
        dh_num = num_grad(fh_h, prev_h, dnext_h) + num_grad(fh_c, prev_h, dnext_c)
        dc_num = num_grad(fc_h, prev_c, dnext_h) + num_grad(fc_c, prev_c, dnext_c)
        dWx_num = num_grad(fWx_h, Wx, dnext_h) + num_grad(fc_c, Wx, dnext_c)
        dWh_num = num_grad(fWh_h, Wh, dnext_h) + num_grad(fWh_c, Wh, dnext_c)
        db_num = num_grad(fb_h, b, dnext_h) + num_grad(fb_c, b, dnext_c)

        dx, dh, dc, dWx, dWh, db = rnn_layers.lstm_step_backward(dnext_h, dnext_c, cache)

        # Compute errors
        err = {}
        err['dx_err']  = error.rel_error(dx, dx_num)
        err['dh_err']  = error.rel_error(dh, dh_num)
        err['dc_err']  = error.rel_error(dc, dc_num)
        err['dWx_err'] = error.rel_error(dWx, dWx_num)
        err['dWh_err'] = error.rel_error(dWh, dWh_num)

        for k, v in err.items():
            print('%s' % k)
            self.assertLessEqual(v, self.eps)
            print("%s err : %f" % (k, v))

        print("\n======== TestCaptioningLSTM.test_lstm_step_backward: <END>")

if __name__ == '__main__':
    unittest.main()
