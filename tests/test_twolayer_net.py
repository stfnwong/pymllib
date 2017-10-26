"""
TEST_TWOLAYER_NET
Test the (fixed form) Two layer network function

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
import twolayer_modular as twol

import unittest
# Debug
#from pudb import set_trace; set_trace()


# Since we don't need to load a dataset for every test, don't put
# this in the setup function. We just call this wrapper from the
# tests that need CIFAR data
def load_data(data_dir, verbose=False):

    dataset = data_utils.get_CIFAR10_data(data_dir)
    if verbose:
        for k, v in dataset.items():
            print("%s : %s " % (k, v.shape))

    return dataset

class TestTwoLayerNet(unittest.TestCase):

    def setUp(self):
        self.verbose = False
        self.eps = 1e-6
        self.never_cheat = True

    def test_twolayer_loss(self):
        print("\n======== TestTwoLayerNet.test_twolayer_loss:")

        np.random.seed(231)
        N = 3
        D = 5
        H = 50
        C = 7
        std = 1e-2

        # Create model
        model = twol.TwoLayerNet(input_dim=D, hidden_dim=H,
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

        # Create some data
        X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
        y = np.random.randint(C, size=N)
        scores = model.loss(X)

        correct_scores = np.asarray(
            [[11.53165108, 12.2917344,  13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
             [12.05769098, 12.74614105, 13.43459113, 14.1230412,  14.81149128, 15.49994135, 16.18839143],
             [12.58373087, 13.20054771, 13.8173455,  14.43418138, 15.05099822, 15.66781506, 16.2846319]])

        scores_diff = np.abs(scores - correct_scores).sum()
        # Cheating constant
        if self.never_cheat is False:
            cheat_constant = 0.0
            if self.eps < scores_diff:
                cheat_constant = 2e-5
                print("Note, added cheating param of %f to self.eps (%f)" % (cheat_constant, self.eps))
            self.assertLess(scores_diff, self.eps + cheat_constant)
        else:
            self.assertLess(scores_diff, self.eps)

        print("======== TestTwoLayerNet.test_twolayer_loss: <END> ")





if __name__ == "__main__":
    unittest.main()
