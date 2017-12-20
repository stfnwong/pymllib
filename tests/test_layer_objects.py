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
        self.eps = 1e-4
        self.verbose = True

    def test_affine_forward(self):
        print("======== TestAffineLayerObject.test_affine_forward:")

        weight_scale = 1e-3
        weight_init = 'gauss_sqrt'
        N = 64
        D = 150
        affine_layer = layer_objects.AffineLayer(weight_scale, weight_init, N, D)

        print(affine_layer)
        self.assertEqual(affine_layer.W.shape[0], N)
        self.assertEqual(affine_layer.W.shape[1], D)

        X = np.random.randn(D, N)
        affine_layer.forward(X)
        print(affine_layer)

        print("======== TestAffineLayerObject.test_affine_forward: <END> ")


    def test_affine_backward(self):
        print("======== TestAffineLayerObject.test_affine_backward:")

        weight_scale = 1e-3
        weight_init = 'gauss_sqrt'
        N = 64
        D = 150
        affine_layer = layer_objects.AffineLayer(weight_scale, weight_init, N, D)

        print(affine_layer)
        self.assertEqual(affine_layer.W.shape[0], N)
        self.assertEqual(affine_layer.W.shape[1], D)

        print('Computing affine forward pass')
        X = np.linspace(-0.5, 0.5, num=N*D).reshape(D, N)
        h = affine_layer.forward(X)
        print('forward activation shape: %s' % str(h.shape))
        print('Computing affine backward pass')
        #affine_layer.backward()
        dz = np.random.randn(*X.shape)
        print('Gradient shape : %s' % str(dz.shape))
        dx = affine_layer.backward(dz)
        fx = lambda x: affine_layer.backward(dz)[0]
        dx_num = check_gradient.eval_numerical_gradient_array(fx, X, dz)
        dx_err = error.rel_error(dx, dx_num)

        print('dx error: %f' % dx_err)
        self.assertLessEqual(dx_err, self.eps)

        print("======== TestAffineLayerObject.test_affine_backward: <END> ")



if __name__ == '__main__':
    unittest.main()
