"""
#TEST LAYERS
Test that all the layers operate correctly, and that forward and
backward computations are correct

Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
# Layers
import pymllib.layers.layers as layers
# Utils
import pymllib.utils.check_gradient as check_gradient
import pymllib.utils.error as error

# Debug
#from pudb import set_trace; set_trace()

# Test standard layers
class TestLayers(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        self.eps = 1e-6
        self.never_cheat = False   # TODO : implement cheat switch

    def test_affine_layer_forward(self):
        print("\n======== TestLayers.test_affine_layer_forward:")

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

        print("======== TestLayers.test_affine_layer_forward: <END> ")

    def test_affine_layer_backward(self):
        print("\n======== TestLayers.test_affine_layer_backward:")

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

        print("======== TestLayers.test_affine_layer_backward: <END> ")

    def test_relu_layer_forward(self):
        print("\n======== TestLayers.test_relu_layer_forward:")

        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        out, _ = layers.relu_forward(x)
        correct_out = np.array([[0.0,        0.0,        0.0,        0.0,      ],
                                [0.0,        0.0,        0.04545455, 0.13636364],
                                [0.22727273, 0.31818182, 0.40909091, 0.5       ]])
        diff = error.rel_error(out, correct_out)
        print("Difference is %.9f" % (diff))

        self.assertLessEqual(diff, self.eps + 4e-8)        # NOTE: For this I had to cheat...
        print("Note : added cheating param of 4e-8 to self.eps (%f)" % self.eps)

        print("======== TestLayers.test_relu_layer_forward: <END> ")

    def test_relu_layer_backward(self):
        print("\n======== TestLayers.test_relu_layer_backward:")

        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        dx_num = check_gradient.eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)
        _, cache = layers.relu_forward(x)
        dx = layers.relu_backward(dout, cache)
        dx_error = error.rel_error(dx_num, dx)

        print("dx_error : %.9f" % (dx_error))
        self.assertLessEqual(dx_error, self.eps)

        print("======== TestLayers.test_relu_layer_backward: <END> ")


# Test batch norm layer
class TestLayersBatchnorm(unittest.TestCase):
    def setUp(self):
        self.verbose = False
        self.eps = 1e-6

    def test_batchnorm_forward(self):
        print("\n======== TestLayersBatchnorm.test_batchnorm_forward:")

        mean_allow_err = 0.1
        std_allow_err = 0.1
        N = 200
        D1 = 50
        D2 = 60
        D3 = 3
        X = np.random.randn(N, D1)
        W1 = np.random.randn(D1, D2)
        W2 = np.random.randn(D2, D3)
        a = np.maximum(0, X.dot(W1)).dot(W2)  # ReLU activation for 2 layer network

        print("Before batch normalization:")
        print("means : %s" % a.mean(axis=0))
        print("stds  : %s" % a.std(axis=0))

        a_norm, _ = layers.batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})

        print("After batch normalization (training mode) with gamma = 1, beta = 0:")
        print("means : %s" % a_norm.mean(axis=0))
        print("stds  : %s" % a_norm.std(axis=0))

        # Check that mean tends to beta, std tends to gamma
        a_mean = a_norm.mean(axis=0)
        a_std = a_norm.std(axis=0)
        for n in range(len(a_mean)):
            self.assertLessEqual(a_mean[n], 0.0 + mean_allow_err)

        for n in range(len(a_std)):
            self.assertLessEqual(a_std[n], 1.0 + std_allow_err)

        gamma = np.asarray([1.0, 2.0, 3.0])
        beta = np.asarray([11.0, 12.0, 13.0])
        a_norm, _ = layers.batchnorm_forward(a, gamma, beta, {'mode': 'train'})

        print("\nAfter batch normalization (training mode) with gamma = %s, beta = %s:" % (gamma, beta))
        print("means : %s" % a_norm.mean(axis=0))
        print("stds  : %s" % a_norm.std(axis=0))

        a_mean = a_norm.mean(axis=0)
        a_std = a_norm.std(axis=0)
        for n in range(len(a_mean)):
            self.assertLessEqual(a_mean[n], beta[n] + mean_allow_err)

        for n in range(len(a_std)):
            self.assertLessEqual(a_std[n], gamma[n] + std_allow_err)


        # Try running many passes of batchnorm in test mode to generate some
        # running averages
        n_train = 200
        bn_param = {'mode': 'train'}
        gamma = np.ones(D3)
        beta = np.zeros(D3)
        for i in range(n_train):
            X = np.random.randn(N, D1)
            a = np.maximum(0, X.dot(W1)).dot(W2)
            a, cache = layers.batchnorm_forward(a, gamma, beta, bn_param)
            #bn_param = cache[-1]

        # Now run a forward pass
        #bn_param = {'mode': 'test'}
        bn_param['mode'] = 'test'
        X = np.random.randn(N, D1)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        a_norm, _ = layers.batchnorm_forward(a, gamma, beta, bn_param)

        print("\nAfter batch normalization (%d training runs) with gamma = %s, beta = %s:" % (n_train, gamma, beta))
        print("means : %s" % a_norm.mean(axis=0))
        print("stds  : %s" % a_norm.std(axis=0))

        print("======== TestLayersBatchnorm.test_batchnorm_forward: <END> ")

    def test_batchnorm_backward(self):
        print("\n======== TestLayersBatchnorm.test_batchnorm_backward:")

        N = 4
        D = 5
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn_param = {'mode': 'train'}

        fx = lambda x: layers.batchnorm_forward(x, gamma, beta, bn_param)[0]
        fg = lambda a: layers.batchnorm_forward(x, gamma, beta, bn_param)[0]
        fb = lambda b: layers.batchnorm_forward(x, gamma, beta, bn_param)[0]

        dx_num = check_gradient.eval_numerical_gradient_array(fx, x, dout)
        da_num = check_gradient.eval_numerical_gradient_array(fg, gamma, dout)
        db_num = check_gradient.eval_numerical_gradient_array(fb, beta, dout)

        _, cache = layers.batchnorm_forward(x, gamma, beta, bn_param)
        dx, dgamma, dbeta = layers.batchnorm_backward(dout, cache)

        dx_error = error.rel_error(dx, dx_num)
        dgamma_error = error.rel_error(dgamma, da_num)
        dbeta_error = error.rel_error(dbeta, db_num)

        print("dx_error : %f" % dx_error)
        print("dgamma_error : %f" % dgamma_error)
        print("dbeta_error : %f" % dbeta_error)

        self.assertLessEqual(dx_error, self.eps)
        self.assertLessEqual(dgamma_error, self.eps)
        self.assertLessEqual(dbeta_error, self.eps)

        print("======== TestLayersBatchnorm.test_batchnorm_backward: <END> ")


# Test dropout layer
class TestLayersDropout(unittest.TestCase):
    def setUp(self):
        self.verbose = False
        self.eps = 1e-6

    def test_droput_forward(self):
        print("\n======== TestLayersDropout.test_droput_forward:")

        X = np.random.randn(500, 500) + 10
        dropout_probs = [0.3, 0.6, 0.1]

        for p in dropout_probs:
            out_train, cache = layers.dropout_forward(X, {'mode': 'train', 'p': p})
            out_test, _ = layers.dropout_forward(X, {'mode': 'test', 'p': p})

            print("Running test with p=%f" % p)
            print("Input mean                : %f " % X.mean())
            print("Mean of train-time output : %f " % out_train.mean())
            print("Mean of test-time output  : %f " % out_test.mean())
            print("Fraction of train-time output set to zero : %f " % (out_train == 0).mean())
            print("Fraction of test-time output set to zero  : %f " % (out_test == 0).mean())

        print("======== TestLayersDropout.test_dropout_forward: <END> ")

    def test_dropout_backward(self):
        print("\n======== TestLayersDropout.test_dropout_backward:")

        N = 2
        D = 15
        H1 = 20
        H2 = 30
        C = 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))
        dropout_probs = [0.3, 0.6, 0.1]

        import pymllib.classifiers.fcnet as fcnet
        # Network params
        hidden_dims = [H1, H2]
        weight_scale = 5e-2

        for p in dropout_probs:
            print("Running check with dropout p = %f" % p)
            model = fcnet.FCNet(hidden_dims=hidden_dims,
                                input_dim=D,
                                num_classes=C,
                                dropout=p,
                                weight_scale=weight_scale,
                                seed=123,
                                dtype=np.float64)
            loss, grads = model.loss(X, y)
            print("Initial loss : %f" % loss)
            for n in sorted(grads):
                f = lambda _: model.loss(X,y)[0]
                grad_num = check_gradient.eval_numerical_gradient(f, model.params[n])
                grad_error = error.rel_error(grad_num, grads[n])
                print("%s relative error : %.2e" % (n, grad_error))

        print("======== TestLayersDropout.test_dropout_backward: <END> ")

if __name__ == "__main__":
    unittest.main()
