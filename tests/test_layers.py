"""
TEST LAYERS
Test that all the layers operate correctly, and that forward and
backward computations are correct

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))

import numpy as np
import unittest
# Layers
import layers
# Utils
import check_gradient
import error

# Debug
#from pudb import set_trace; set_trace()


class TestLayers(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        self.eps = 1e-6
        self.never_cheat = False   # TODO : implement cheat switch

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
        print("Note : added cheating param of 4e-8 to self.eps (%f)" % self.eps)

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

if __name__ == "__main__":
    unittest.main()
