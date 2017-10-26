"""
TEST_SOLVER
Test the solver object and the various optimization functions
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../solver')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifiers')))

import numpy as np
import error
# Units under test
import solver
import optim
import fcnet

import unittest



class TestSolver(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-6

    def test_sgd_momentum(self):
        print("\n======== TestSolver.test_sgd_momentum:")

        N = 4
        D = 5

        w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
        dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
        v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)

        config = {'learning_rate': 1e-3, 'velocity': v}
        next_w, _ = optim.sgd_momentum(w, dw, config=config)
        expected_next_w = np.asarray([
        [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
        [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
        [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
        [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
        expected_velocity = np.asarray([
        [ 0.5406,      0.55475789,  0.56891579,  0.58307368,  0.59723158],
        [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
        [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
        [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])

        next_w_error = error.rel_error(next_w, expected_next_w)
        velocity_error = error.rel_error(config['velocity'], expected_velocity)

        print("next_w_error = %f" % next_w_error)
        print("velocity_error = %f" % velocity_error)

        self.assertLessEqual(next_w_error, self.eps)
        self.assertLessEqual(velocity_error, self.eps)





if __name__ == "__main__":
    unittest.main()
