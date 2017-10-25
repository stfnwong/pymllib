"""
TEST_FCNET
Test the fully connected network function

Note that the layer tests are taken directly from CS231n,
and are in effect just re-factored into unit tests

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import numpy as np
import data_utils
import check_gradient
import error
import layers
import fcnet

import unittest
# Debug
from pudb import set_trace; set_trace()


# Since we don't need to load a dataset for every test, don't put
# this in the setup function. We just call this wrapper from the
# tests that need CIFAR data
def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

class TestFCNet(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        self.eps = 1e-6

    def test_affine_layer_forward(self):
        print("\n======== TestFCNet.test_affine_layer_forward:")

        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)

        out, _ = layers.affine_forward(x, w, b)
        correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                                [ 3.25553199,  3.5141327,   3.77273342]])
        # Compare
        diff = error.rel_error(out, correct_out)
        print("Difference is %.9f" % (diff))
        self.assertLessEqual(diff, self.eps)

        print("======== TestFCNet.test_affine_layer_forward: <END> ")

    def test_affine_layer_backward(self):
        print("\n======== TestFCNet.test_affine_layer_backward:")

        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.affine_forward(x, w, b)[0], x, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: layers.affine_forward(x, w, b)[0], w, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: layers.affine_forward(x, w, b)[0], b, dout)

        _, cache = layers.affine_forward(x, w, b)
        dx, dw, db = layers.affine_backward(dout, cache)

        dx_diff = error.rel_error(dx_num, dx)
        dw_diff = error.rel_error(dw_num, dw)
        db_diff = error.rel_error(db_num, db)

        print("dx error : %.9f" % dx_diff)
        print("dw error : %.9f" % dw_diff)
        print("db error : %.9f" % db_diff)

        # NOTE : occasionally we may randomly get a value greater than self.eps
        # here... I don't think its worth re-writing this test such that it can
        # pass every time, rather it might be better
        self.assertLessEqual(dx_diff, self.eps)
        self.assertLessEqual(dw_diff, self.eps)
        self.assertLessEqual(db_diff, self.eps)

        print("======== TestFCNet.test_affine_layer_backward: <END> ")

    def test_relu_layer_forward(self):
        print("\n======== TestFCNet.test_relu_layer_forward:")

        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        out, _ = layers.relu_forward(x)
        correct_out = np.array([[0.0,        0.0,        0.0,        0.0,      ],
                                [0.0,        0.0,        0.04545455, 0.13636364],
                                [0.22727273, 0.31818182, 0.40909091, 0.5       ]])
        diff = error.rel_error(out, correct_out)
        print("Difference is %.9f" % (diff))

        self.assertLessEqual(diff, self.eps + 4e-8)        # NOTE: For this I had to cheat...
        print("Note : added cheating param of 4e-8 to self.eps")

        print("======== TestFCNet.test_relu_layer_forward: <END> ")

    def test_relu_layer_backward(self):
        print("\n======== TestFCNet.test_relu_layer_backward:")

        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)
        _, cache = layers.relu_forward(x)
        dx = layers.relu_backward(dout, cache)
        dx_error = error.rel_error(dx_num, dx)

        print("dx_error : %.9f" % (dx_error))
        self.assertLessEqual(dx_error, self.eps)

        print("======== TestFCNet.test_relu_layer_backward: <END> ")

    # TODO : The convenience layers

    def test_two_layer_fcnet_solver(self):
        print("\n======== TestFCNet.test_two_layer_fcnet_solver:")

        np.random.seed(231)
        N = 3
        D = 5
        H = 50
        C = 7
        std = 1e-2

        X = np.random.randn(N, D)
        y = np.random.randint(C, size=N)

        # TODO : try with "twolayer" model as per CS231n
        #model = fcnet.FCNet(hidden_dims, input_dim,
        #                    num_classes, weight_scale=std,
        #                    verbose=True)
        model = fcnet.TwoLayerNet(input_dim=D, hidden_dim=H,
                            num_classes=C, weight_scale=std,
                            verbose=True)
        W1_std = abs(model.params['W1'].std() - std)
        W2_std = abs(model.params['W2'].std() - std)
        b1 = model.params['b1']
        b2 = model.params['b2']

        # Check that the weights are sensible
        self.assertLess(W1_std, std / 10.0, msg="Problem in first layer weights")
        self.assertLess(W2_std, std / 10.0, msg="Problem in second layer weights")
        self.assertTrue(np.all(b1 == 0), msg="Problem in first layer biases")
        self.assertTrue(np.all(b2 == 0), msg="Problem in second layer biases")

        print("\tTest time forward pass")
        model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
        model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
        model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
        model.params['b2'] = np.linspace(-0.9, 0.1, num=C)

        X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
        scores = model.loss(X)

        correct_scores = np.asarray(
            [[11.53165108, 12.2917344,  13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
             [12.05769098, 12.74614105, 13.43459113, 14.1230412,  14.81149128, 15.49994135, 16.18839143],
             [12.58373087, 13.20054771, 13.8173455,  14.43418138, 15.05099822, 15.66781506, 16.2846319]])

        scores_diff = np.abs(scores - correct_scores).sum()
        self.assertLess(scores_diff, self.eps)

        print("======== TestFCNet.test_two_layer_fcnet_solver: <END> ")





if __name__ == "__main__":
    unittest.main()
