"""
TEST_CL_UTIL
Unit tests for OpenCL utilities.


Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
# Module under test
from pymllib.opencl import cl_util

# Debug
from pudb import set_trace; set_trace()





class TestCLUtil(unittest.TestCase):

    def setUp(self):
        self.verbose = True
        self.cl_platform_string = 'Intel Gen OCL Driver'

    #def test_get_ctx(self):
    #    print("\n======== TestCLUtil.test_get_ctx:")
    #    print("======== TestCLUtil.test_get_ctx: <END> ")

    def test_get_platform_info(self):

        platform = cl_util.cl_select_platform(self.cl_platform_string)
        if platform is None:
            print("Failed to find platform")
        else:
            print(platform)


if __name__ == "__main__":
    unittest.main()
