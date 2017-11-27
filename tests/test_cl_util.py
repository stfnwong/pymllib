"""
TEST_CL_UTIL
Unit tests for OpenCL utilities.


Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pyopencl as cl
# Module under test
from pymllib.opencl import cl_util

# Debug
#from pudb import set_trace; set_trace()

class TestCLUtil(unittest.TestCase):

    def setUp(self):
        self.verbose = True
        self.cl_platform_string = 'Intel Gen OCL Driver'


    def test_init_context(self):
        print("\n======== TestCLUtil.test_init_context:")

        cl_context = cl_util.clContext(verbose=self.verbose)
        try:
            cl_context.init_context()
        except ValueError as e:
            print(e)
            print("Failed test_init_context")
            return

        # Ensure that members have values
        self.assertIsNotNone(cl_context.context)
        self.assertIsNotNone(cl_context.device)


        print("======== TestCLUtil.test_init_context: <END> ")



    #def test_load_kernel(self):
    #    print("\n======== TestCLUtil.test_load_kernel:")

    #    # TODO: Using the Intel driver for testing, change
    #    # to AMD driver on workstation
    #    preferred_platform = 'Intel Gen OCL Driver'
    #    preferred_device_type = 'GPU'

    #    # Get a context object
    #    context = cl_util.clContext(platform_str = preferred_platform,
    #                      device_type = preferred_device_type,
    #                      verbose = self.verbose)
    #    test_kernel = 'pymllib/opencl/kernels/sgemm_test.cl'
    #    context.load_kernel(test_kernel)
    #    # Print the kernels that we loaded
    #    print("Context contains the following kernels")
    #    for k in context.kernels:
    #        print(str(k))


    #    print("======== TestCLUtil.test_load_kernel: <END> ")


if __name__ == "__main__":
    unittest.main()
