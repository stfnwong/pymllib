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

def read_source(filename):
    with open(filename, 'r') as fp:
        source = fp.read().rstrip('\n')

    return source

class TestCLProgram(unittest.TestCase):
    def setUp(self):
        self.verbose = True
        #self.cl_platform_string = 'Intel Gen OCL Driver'
        self.cl_platform_string = 'AMD Accelerated Parallel Processing'
        #self.kernel_source = 'pymllib/opencl/kernels/sum.cl'
        self.kernel_file = 'pymllib/opencl/kernels/sgemm.cl'
        self.kernel_name = 'sgemm_naive'
        self.dtype = np.float32

    def test_sgemm_kernels(self):
        print("\n======== TestCLProgram.test_sgemm_kernels:")

        kernel_names = ['sgemm_naive', 'sgemm_tiling']
        # Get source
        source = read_source(self.kernel_file)
        # Get dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        # Get a program object
        cl_program = cl_util.clProgram(verbose=self.verbose)
        kernels = cl_program.build(ctx, source, device=device)
        print("Built %d kernel(s)" % len(kernels.keys()) )
        for k, v in kernels.items():
            print('\t%s : %s' % (k, v))

        for n in kernel_names:
            self.assertTrue(n in kernels.keys())



        print("======== TestCLProgram.test_sgemm_kernels: <END> ")

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
        print("Reading source file from %s" % self.kernel_file)
        source = read_source(self.kernel_file)
        # Build the kernels in the source file
        kernels = cl_program.build(ctx, source, device=device)
        print("Built %d kernel(s)" % len(kernels.keys()) )
        for k, v in kernels.items():
            print('\t%s : %s' % (k, v))

        self.assertTrue(self.kernel_name in kernels.keys())

        # Create some dummy data
        print("Generating test data...")
        A = np.linspace(1, 64, num=64*64).astype(self.dtype)
        A = A.reshape((64,64))
        B = np.linspace(1, 64, num=64*64).astype(self.dtype)
        B = B.reshape((64,64))
        print('A shape : %s' % str(A.shape))
        print('B shape : %s' % str(B.shape))
        #A = np.random.randn(64, 64).astype(self.dtype)
        #B = np.random.randn(64, 64).astype(self.dtype)
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        r_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)
        # Note, we must use np.int32's here to ensure the correct alignment
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[0])
        K = np.int32(A.shape[0])
        # Set the kernel args
        kernels[str(self.kernel_name)].set_args(M, N, K, a_buf, b_buf, r_buf)
        # Stick the kernel in the queue
        ev = cl.enqueue_nd_range_kernel(queue, kernels[str(self.kernel_name)], A.shape, None)
        print(ev)
        # Results
        C = np.dot(A, B)
        cl_result = np.empty_like(C).astype(self.dtype)
        cl.enqueue_copy(queue, cl_result, r_buf)

        diff = abs(C - cl_result)
        print("C = A * B : \n")
        print(C)
        print("CL result :\n")
        print(cl_result)
        print('max diff')
        print(np.max(diff))

        print("======== TestCLProgram.test_build_program: <END> ")







if __name__ == "__main__":
    unittest.main()
