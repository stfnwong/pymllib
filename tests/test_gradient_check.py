"""
TEST_GRADIENT_CHECK

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import pymllib.layers.layers as layers
import pymllib.utils.check_gradient as check_gradient
import pymllib.utils.error as error

# Debug
#from pudb import set_trace; set_trace()

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class TestGradientCheck(unittest.TestCase):

    def setUp(self):
        pass

    def test_gradient(self):

        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)

        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.affine_forward(x, w, b)[0], x, dout)
        dw_num = check_gradient.eval_numerical_gradient_array(lambda w: layers.affine_forward(x, w, b)[0], w, dout)
        db_num = check_gradient.eval_numerical_gradient_array(lambda b: layers.affine_forward(x, w, b)[0], b, dout)

        _, cache = layers.affine_forward(x, w, b)
        dx, dw, db = layers.affine_backward(dout, cache)

        print("dx error : %.6f " % error.rel_error(dx_num, dx))
        print("dw error : %.6f " % error.rel_error(dw_num, dw))
        print("db error : %.6f " % error.rel_error(db_num, db))

if __name__ == "__main__":
    unittest.main()
