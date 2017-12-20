"""
TEST_LAYER_OBJECTS

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import numpy as np
# Internal modules
from pymllib.layers import layer_objects
from pymllib.utils import check_gradient
from pymllib.utils import error

# Debug
#from pudb import set_trace; set_trace()


class TestAffineLayerObject(unittest.TestCase):
    def setUp(self):
        self.weight_scale = 1e-2
        self.weight_init = 'gauss_sqrt'
        self.eps = 1e-4
        self.verbose = True

    def test_affine_forward(self):
        print("======== TestAffineLayerObject.test_affine_forward:")

        N = 4
        D = 4
        affine_layer = layer_objects.AffineLayer(
            self.weight_scale, self.weight_init, N, D)

        print(affine_layer)
        self.assertEqual(affine_layer.W.shape[0], N)
        self.assertEqual(affine_layer.W.shape[1], D)

        X = np.random.randn(D, N)
        h = affine_layer.forward(X)
        print(affine_layer)
        expected_h = np.asarray([[-0.02388677,  0.05494526, -0.05156986,  0.01443091],
                      [-0.0001368 , -0.00902937,  0.04242849, -0.01104825],
                      [-0.01300566,  0.00471226, -0.00942038,  0.01923142],
                      [ 0.03432371, -0.02492158,  0.04154934, -0.06043238]])
        h_err = error.rel_error(h, expected_h)
        print('h_err : %f' % h_err)
        self.assertLessEqual(h_err, self.eps)

        print("======== TestAffineLayerObject.test_affine_forward: <END> ")


    def test_affine_backward(self):
        print("======== TestAffineLayerObject.test_affine_backward:")
        N = 4
        D = 8
        affine_layer = layer_objects.AffineLayer(
            self.weight_scale, self.weight_init, N, D)

        print(affine_layer)
        self.assertEqual(affine_layer.W.shape[0], N)
        self.assertEqual(affine_layer.W.shape[1], D)

        print('Computing affine forward pass')
        X = np.linspace(-0.5, 0.5, num=N*D).reshape(N, D)
        print('X shape : %s' % (str(X.shape)))
        h = affine_layer.forward(X)
        print('forward activation shape: %s' % str(h.shape))
        print('Computing affine backward pass')
        #affine_layer.backward()
        dz = np.random.randn(*X.shape)
        print('Gradient shape : %s' % str(dz.shape))
        dx = affine_layer.backward(dz)
        #fx = lambda x: affine_layer.backward(dz)[0]
        dx_num = check_gradient.eval_numerical_gradient_array(
            lambda x: affine_layer.backward(dz)[0], X, dz)
        dx_err = error.rel_error(dx, dx_num)

        print('dx error: %f' % dx_err)
        self.assertLessEqual(dx_err, self.eps)

        print("======== TestAffineLayerObject.test_affine_backward: <END> ")

    def test_relu_forward(self):
        print("======== TestAffineLayerObject.test_relu_forward:")

        X = np.random.randn(10, 10)
        dout = np.random.randn(*X.shape)
        relu_layer = layer_objects.ReLULayer(
            self.weight_scale, self.weight_init, 10, 10)

        h = relu_layer.forward(X)


        print("======== TestAffineLayerObject.test_relu_forward: <END> ")

    def test_relu_backward(self):
        print("======== TestAffineLayerObject.test_relu_backward:")

        X = np.random.randn(10, 10)
        dout = np.random.randn(*X.shape)
        relu_layer = layer_objects.ReLULayer(
            self.weight_scale, self.weight_init, 10, 10)
        relu_layer.X = X    # store cache


        dx_num = check_gradient.eval_numerical_gradient_array(
            lambda x: relu_layer.backward(dout)[0], X, dout)
        dx = relu_layer.backward(dout)
        dx_error = error.rel_error(dx_num, dx)

        print("dx_error : %.9f" % (dx_error))
        self.assertLessEqual(dx_error, self.eps)

        print("======== TestAffineLayerObject.test_relu_backward: <END> ")

if __name__ == '__main__':
    unittest.main()
