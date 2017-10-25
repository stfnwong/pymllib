"""
TEST_FCNET
Test the fully connected network function

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../layers')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))

import numpy as np
import data_utils
import check_gradient
import layers

import unittest
# Debug
#from pudb import set_trace; set_trace()

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class TestFCNet(unittest.TestCase):

    def setUp(self):
        data_dir = 'datasets/cifar-10-batches-py/'

        print("==== Loading data from %s" % (data_dir))
        self.dataset = data_utils.get_CIFAR10_data(data_dir)
        print("=== Loaded data...")
        for k, v in self.dataset.items():
            print("%s : %s " % (k, v.shape))

    def test_affine_layer_forward(self):
        print("======== TestFCNet.test_affine_layer_forward:")

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
        diff = rel_error(out, correct_out)
        print("Difference is %f" % (diff))
        print("output shape : (%d, %d)" % (out.shape[0], out.shape[1]))
        self.assertLessEqual(out, correct_out)

        print("======== TestFCNet.test_affine_layer: <END> ")

    def test_affine_layer_backward(self):
        print("======== TestFCNet.test_affine_layer_backward:")

        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.affine_forward(x, w, b)[0], x, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: layers.affine_forward(x, w, b)[0], w, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: layers.affine_forward(x, w, b)[0], b, dout)

        _, cache = layers.affine_forward(x, w, b)
        dx, dw, db = layers.affine_backward(dout, cache)

        print("dx error : %f" % (rel_error(dx_num, dx)))
        print("dw error : %f" % (rel_error(dw_num, dw)))
        print("db error : %f" % (rel_error(db_num, db)))

        # TODO : Asserts?




if __name__ == "__main__":
    unittest.main()
