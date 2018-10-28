"""
TEST_AUTO_TEST
Test harness for the sample autoencoder

Stefan Wong 2018
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import unittest

import pymllib.utils.data_utils as data_utils
import pymllib.classifiers.autoencoder as autoencoder
import pymllib.classifiers.auto_test as auto_test
import pymllib.solver.solver as solver

# Debug
from pudb import set_trace; set_trace()


class TestAutoTest(unittest.TestCase):
    def setUp(self):
        self.verbose = True

    def test_init(self):
        print("======== TestAutoTest.test_init: ")

        auto = auto_test.AutoTest(verbose=self.verbose)

        print("======== TestAutoTest.test_init: <END> ")


if __name__ == '__main__':
    unittest.main()
