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
import numpy as np
# Module under test
from pymllib.opencl import cl_util

# Debug
#from pudb import set_trace; set_trace()


def create_cl_test_harness(platform_str='AMD'):
    """
    CREATE_CL_TEST_HARNESS
    Utility function to create a useable cl context
    """
    # Get a platform
    platform = None
    platform_list = cl.get_platforms()
    for p in platform_list:
        if platform_str in p.name:
            platform = p
            break
    if platform is None:
        if len(platform_list) >= 1:
            platform = platform_list[0]
        else:
            print("Failed to get platform")
            return None, None, None, None

    print('Platform name is %s' % platform.name)
    print('Platform vendor is %s' % platform.vendor)

    # Get a device
    device = None
    device_list = platform.get_devices()
    for d in device_list:
        # prefer a GPU
        if cl.device_type.to_string(d.type) == 'GPU':
            device = d
    if device is None:
        if len(device_list) >= 1:
            device = device_list[0]
        else:
            print("Failed to get device")
            return None, None, None, None

    print('Device name is %s' % device.name)
    print('Device type is %s' % cl.device_type.to_string(device.type))

    # Create a context`
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    return ctx, queue, platform, device



class TestCLProgram(unittest.TestCase):
    def setUp(self):
        self.verbose = True
        #self.cl_platform_string = 'Intel Gen OCL Driver'
        self.cl_platform_string = 'AMD Accelerated Parallel Processing'
        self.kernel_source = 'pymllib/opencl/kernels/sgemm_test.cl'
        self.dtype = np.float32

    def test_build_program(self):
        print("\n======== TestCLProgram.test_build_program:")
        # Create dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        if ctx is None:
            print('Failed to init test harness')
            return
        # Get a program object
        cl_program = cl_util.clProgram()
        # Get some source
        with open(self.kernel_source, 'r') as fp:
            source = fp.read().replace('\n', '')

        cl_program.build(ctx, source, device=device)
        for k, v in cl_program.kernels.items():
            print('%s : %s' % (k, v))

        # Create some dummy data
        print("Generating test data...")
        A = np.random.randn(64, 64).astype(self.dtype)
        B = np.random.randn(64, 64).astype(self.dtype)
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        result = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)


        print("======== TestCLProgram.test_build_program: <END> ")

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
        self.assertIsNotNone(cl_context.queue)
        self.assertIsNotNone(cl_context.platform)
        self.assertIsNotNone(cl_context.device)


        print("======== TestCLUtil.test_init_context: <END> ")

    def test_build_program(self):
        print("\n======== TestCLUtil.test_build_program:")
        test_kernel_source_file = 'pymllib/opencl/kernels/sgemm_test.cl'

        cl_context = cl_util.clContext(verbose=self.verbose)
        try:
            cl_context.init_context()
        except ValueError as e:
            print(e)
            print("Failed test_init_context")
            return

        cl_context.prog_from_file(test_kernel_source_file)
        print(cl_context)


        print("======== TestCLUtil.test_build_program: <END> ")




if __name__ == "__main__":
    unittest.main()
