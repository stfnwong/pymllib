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
from pudb import set_trace; set_trace()


class TestCLKernel(unittest.TestCase):

    def setUp(self):
        self.verbose = True

    def test_load_kernel(self):
        print("\n======== TestCLUtil.test_load_kernel:")

        os.environ['PYOPENCL_CTX'] = '0'

        # get some context for the test harness
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        platform_list = cl.get_platforms()
        platform = platform_list[0]

        # TODO: handle exceptions
        device_list = platform.get_devices()
        device = device_list[0]

        # get the kernel object
        test_kernel = 'pymllib/opencl/kernels/sgemm_test.cl'
        kernel = cl_util.clKernel()
        kernel.load_source(test_kernel)
        print(kernel)




        print("======== TestCLUtil.test_load_kernel: <END> ")

class TestCLUtil(unittest.TestCase):

    def setUp(self):
        self.verbose = True
        self.cl_platform_string = 'Intel Gen OCL Driver'

    def test_get_platform_info(self):

        platform = cl_util.cl_select_platform(self.cl_platform_string)
        if platform is None:
            print("Failed to find platform")
        else:
            print(platform)

    def test_load_kernel(self):
        print("\n======== TestCLUtil.test_load_kernel:")

        # TODO: Using the Intel driver for testing, change
        # to AMD driver on workstation
        preferred_platform = 'Intel Gen OCL Driver'
        preferred_device_type = 'GPU'

        # Get a context object
        context = cl_util.clContext(platform_str = preferred_platform,
                          device_type = preferred_device_type,
                          verbose = self.verbose)
        test_kernel = 'pymllib/opencl/kernels/sgemm_test.cl'
        context.load_kernel(test_kernel)
        # Print the kernels that we loaded
        print("Context contains the following kernels")
        for k in context.kernels:
            print(str(k))


        print("======== TestCLUtil.test_load_kernel: <END> ")


if __name__ == "__main__":
    unittest.main()
