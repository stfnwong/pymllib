"""
TEST_CL_UTILS
Unit tests for OpenCL utilities.


Stefan Wong 2017
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pyopencl as cl
import numpy as np
import cProfile

# Module under test
from pymllib.opencl import cl_utils

# Debug
#from pudb import set_trace; set_trace()


def do_cprofile(func):
    """
    Decorator functions for profiler
    """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
        return profiled_func

def create_cl_test_harness(platform_str='AMD'):
    """
    CREATE_CL_TEST_HARNESS
    Utility function to get a usable platform, device, context and queue
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
    """
    Read kernel source from disk
    """
    with open(filename, 'r') as fp:
        source = fp.read().rstrip('\n')

    return source

class TestCLProgram(unittest.TestCase):
    """
    Test the CLProgram object
    """
    def setUp(self):
        self.verbose = True
        #self.cl_platform_string = 'Intel Gen OCL Driver'
        self.cl_platform_string = 'AMD Accelerated Parallel Processing'
        #self.kernel_source = 'pymllib/opencl/kernels/sum.cl'
        self.kernel_file = 'pymllib/opencl/kernels/sgemm.cl'
        self.kernel_name = 'sgemm_naive'
        #self.kernel_name = 'sgemm_tile16'
        self.dtype = np.float32

    def test_sgemm_tile16(self):
        print("\n======== TestCLProgram.test_sgemm_tile16:")

        source = read_source(self.kernel_file)
        # Get dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        # Get a program object
        cl_program = cl_utils.clProgram(verbose=self.verbose)
        kernels = cl_program.build(ctx, source, device=device)
        print("Built %d kernel(s)" % len(kernels.keys()) )
        for k, v in kernels.items():
            print('\t%s : %s' % (k, v))

        # Ensure we built the correct kernel
        self.assertTrue('sgemm_tile16' in kernels.keys())

        # Generate test data
        A = np.linspace(1, 64, num=64*64).astype(self.dtype)
        A = A.reshape((64,64))
        B = np.linspace(1, 64, num=64*64).astype(self.dtype)
        B = B.reshape((64,64))
        print('A shape : %s' % str(A.shape))
        print('B shape : %s' % str(B.shape))
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        r_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)

        M = np.int32(A.shape[0])
        N = np.int32(A.shape[0])
        K = np.int32(A.shape[0])

        # Create the reference result
        C = np.dot(A, B)
        cl_result = np.empty_like(C)

        kernels['sgemm_tile16'].set_args(M, N, K, a_buf, b_buf, r_buf)
        print("Enqueuing sgemm_tile16")
        cl.enqueue_nd_range_kernel(queue, kernels['sgemm_tile16'], A.shape, None)
        cl.enqueue_copy(queue, cl_result, r_buf)
        diff = abs(C - cl_result)
        print('sgemm_tile16 difference matrix')
        print(diff)
        self.assertLessEqual(np.max(diff), 1e-8)
        print("Max difference was %f" % np.max(diff))

        print("======== TestCLProgram.test_sgemm_tile16: <END> ")

    def test_sgemm_tile32(self):
        print("\n======== TestCLProgram.test_sgemm_tile32:")

        source = read_source(self.kernel_file)
        # Get dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        # Get a program object
        cl_program = cl_utils.clProgram(verbose=self.verbose)
        kernels = cl_program.build(ctx, source, device=device)
        print("Built %d kernel(s)" % len(kernels.keys()) )
        for k, v in kernels.items():
            print('\t%s : %s' % (k, v))

        # Ensure we built the correct kernel
        self.assertTrue('sgemm_tile32' in kernels.keys())

        # Generate test data
        A = np.linspace(1, 64, num=64*64).astype(self.dtype)
        A = A.reshape((64,64))
        B = np.linspace(1, 64, num=64*64).astype(self.dtype)
        B = B.reshape((64,64))
        print('A shape : %s' % str(A.shape))
        print('B shape : %s' % str(B.shape))
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        r_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)

        M = np.int32(A.shape[0])
        N = np.int32(A.shape[0])
        K = np.int32(A.shape[0])

        # Create the reference result
        C = np.dot(A, B)
        cl_result = np.empty_like(C)

        kernels['sgemm_tile32'].set_args(M, N, K, a_buf, b_buf, r_buf)
        print("Enqueuing sgemm_tile32")
        cl.enqueue_nd_range_kernel(queue, kernels['sgemm_tile32'], A.shape, None)
        cl.enqueue_copy(queue, cl_result, r_buf)
        diff = abs(C - cl_result)
        print("Kernel %s difference matrix" % k)
        print(diff)
        self.assertLessEqual(np.max(diff), 1e-8)
        print("Max difference was %f" % np.max(diff))

        print("======== TestCLProgram.test_sgemm_tile32: <END> ")

    def test_sgemm_kernels(self):
        print("\n======== TestCLProgram.test_sgemm_kernels:")

        kernel_names = ['sgemm_naive', 'sgemm_tile16', 'sgemm_tile32']
        # Get source
        print("Loading source from file %s" % self.kernel_file)
        source = read_source(self.kernel_file)
        # Get dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        # Get a program object
        cl_program = cl_utils.clProgram(verbose=self.verbose)
        kernels = cl_program.build(ctx, source, device=device)
        print("Built %d kernel(s)" % len(kernels.keys()) )
        for k, v in kernels.items():
            print('\t%s : %s' % (k, v))

        # Ensure that all the expected kernels were built
        for n in kernel_names:
            self.assertTrue(n in kernels.keys())

        # Generate test data
        A = np.linspace(1, 64, num=64*64).astype(self.dtype)
        A = A.reshape((64,64))
        B = np.linspace(1, 64, num=64*64).astype(self.dtype)
        B = B.reshape((64,64))
        print('A shape : %s' % str(A.shape))
        print('B shape : %s' % str(B.shape))
        a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)

        result_buffers = {}
        for k in kernels.keys():
            result_buffers[k] = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)

        #r_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=A.nbytes)
        # Note, we must use np.int32's here to ensure the correct alignment
        M = np.int32(A.shape[0])
        N = np.int32(A.shape[0])
        K = np.int32(A.shape[0])

        # Create the reference result
        C = np.dot(A, B)
        cl_result = np.empty_like(C)

        for k in kernels.keys():
            if k == 'sgemm_tile32':
                print("Enqueuing kernel %s" % k)
                kernels[k].set_args(M, N, K, a_buf, b_buf, result_buffers[k])
                cl.enqueue_nd_range_kernel(queue, kernels[k], A.shape, None)
                #if k == 'sgemm_naive':
                #    cl.enqueue_nd_range_kernel(queue, kernels[k], A.shape, None)
                #elif k == 'sgemm_tile16':
                #    cl.enqueue_nd_range_kernel(queue, kernels[k], A.shape, (16,16))
                #elif k == 'sgemm_tile32':
                #    cl.enqueue_nd_range_kernel(queue, kernels[k], A.shape, (32,32))
                cl.enqueue_copy(queue, cl_result, result_buffers[k])
                diff = abs(C - cl_result)
                print("Kernel %s difference matrix" % k)
                print(diff)
                self.assertLessEqual(np.max(diff), 1e-8)
                print("Max difference was %f" % np.max(diff))

        print("======== TestCLProgram.test_sgemm_kernels: <END> ")

    def test_build_program(self):
        print("\n======== TestCLProgram.test_build_program:")
        # Create dummy vars for test
        ctx, queue, platform, device = create_cl_test_harness(platform_str=self.cl_platform_string)
        if ctx is None:
            print('Failed to init test harness')
            return
        # Get a program object
        cl_program = cl_utils.clProgram()
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



class TestCLContext(unittest.TestCase):
    def setUp(self):
        """
        SETUP NOTES:
        Before executing these tests ensure that the following members are
        set to values that reflect the hardware installed on the test system

        self.cl_platform_string
        self.cl_vendor_string
        self.cl_device_name

        The clContext object will attempt to use the specific platform and/or
        device so long as it is available, and thus it is possible to use this
        unit test to determine whether or not a particular platform is operational
        """
        self.verbose = True
        self.cl_platform_string = 'AMD Accelerated Parallel Processing'
        self.cl_vendor_string = 'Advanced Micro Devices'
        self.cl_device_name = 'Bonaire'
        # Source files used in test
        self.kernel_files = ['pymllib/opencl/kernels/sgemm.cl']
        # Number of kernel routines in each source file
        self.num_kernels = [3]
        self.kernel_names = {0: ['sgemm_naive', 'sgemm_tile16', 'sgemm_tile32']}
        self.dtype = np.float32

    def test_context_setup(self):
        print("======== TestCLContext.test_context_setup:")

        cl_context = cl_utils.clContext(platform_str=self.cl_platform_string,
                                        verbose=self.verbose)
        cl_context.init_context()
        # assert that fields were filled in
        self.assertIsNotNone(cl_context.platform)
        self.assertIsNotNone(cl_context.device)
        self.assertIsNotNone(cl_context.context)
        self.assertIsNotNone(cl_context.queue)
        # assert that fields are correct
        self.assertIn(self.cl_vendor_string, cl_context.platform.vendor)
        self.assertIn(self.cl_platform_string, cl_context.platform.name)

        print("======== TestCLContext.test_context_setup: <END> ")

    def test_single_source_load(self):
        print("======== TestCLContext.test_single_source_load:")

        cl_context = cl_utils.clContext(platform_str=self.cl_platform_string,
                                        verbose=self.verbose)
        cl_context.init_context()
        # assert that fields were filled in
        self.assertIsNotNone(cl_context.platform)
        self.assertIsNotNone(cl_context.device)
        self.assertIsNotNone(cl_context.context)
        self.assertIsNotNone(cl_context.queue)
        # assert that fields are correct
        self.assertIn(self.cl_vendor_string, cl_context.platform.vendor)
        self.assertIn(self.cl_platform_string, cl_context.platform.name)

        # Try to load from a single source file
        num_kernels = cl_context.load_source(self.kernel_files[0])
        self.assertEqual(num_kernels, self.num_kernels[0])
        # Check that the names are correct
        if self.verbose:
            print("Kernels in clContext object:")
            for k, v in cl_context.kernels.items():
                print('\t%s : %s' % (k, v))

        for k, v in cl_context.kernels.items():
            self.assertIn(k, self.kernel_names[0])
            print("Kernel %s found" % k)

        print("======== TestCLContext.test_single_source_load: <END> ")


    def test_single_source_exec(self):
        print("======== TestCLContext.test_single_source_exec:")

        cl_context = cl_utils.clContext(platform_str=self.cl_platform_string,
                                        verbose=self.verbose)
        cl_context.init_context()
        # assert that fields were filled in
        self.assertIsNotNone(cl_context.platform)
        self.assertIsNotNone(cl_context.device)
        self.assertIsNotNone(cl_context.context)
        self.assertIsNotNone(cl_context.queue)
        # assert that fields are correct
        self.assertIn(self.cl_vendor_string, cl_context.platform.vendor)
        self.assertIn(self.cl_platform_string, cl_context.platform.name)

        # Try to load from a single source file
        num_kernels = cl_context.load_source(self.kernel_files[0])
        self.assertEqual(num_kernels, self.num_kernels[0])
        # Check that the names are correct
        if self.verbose:
            print("Kernels in clContext object:")
            for k, v in cl_context.kernels.items():
                print('\t%s : %s' % (k, v))

        for k, v in cl_context.kernels.items():
            self.assertIn(k, self.kernel_names[0])
            print("Kernel %s found" % k)

        # Now attempt to execute a kernel in the list

        print("======== TestCLContext.test_single_source_exec: <END> ")


if __name__ == "__main__":
    unittest.main()
