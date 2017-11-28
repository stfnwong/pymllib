"""
CL_UTIL

Various OpenCL utilities
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyopencl as cl
# Debug
from pudb import set_trace; set_trace()


def cl_load_kernel(filename):
    with open(filename, 'r') as fp:
        kernel_source = fp.read().replace('\n', '')

    return kernel_source

def cl_select_platform(pname, verbose=False):

    platform_list = cl.get_platforms()
    for p in platform_list:
        if p.name == pname:
            return p

    if verbose is True:
        print("Couldn't find platform %s" % pname)

    # return the first platform
    if len(platform_list) >= 1:
        return platform_list[0]

    return None

def cl_get_device_info(cl_platform):
    device_list = [device for device in cl_platform.get_devices()]
    return device_list


class clKernel(object):
    """
    clKernel

    Holds a single kernel object extracted from a clProgram
    """
    def __init__(self):
        self.source_file = ''
        self.name = ''
        self.kernel = None

    def __str__(self):
        s = []
        s.append('Name        : %s\n' % self.name)
        s.append('Source File : %s\n' % self.source_file)

        return ''.join(s)

    def __repr__(self):
        return ''.join('%s\n' % self.name)


class clProgram(object):
    def __init__(self):
        self.prog = None
        self.kernels = {}
        self.verbose = True

    def build(self, ctx, source, device=None, options=None):
        self.prog = cl.Program(ctx, source, options)
        self.prog.build()

        # TODO : How to get build errors from here?
        kernel_list = self.prog.all_kernels()
        for k in kernel_list:
            self.kernels[k] = prg.k     # Not sure this is correct....

        # TODO : Debug, remove
        if self.verbose:
            for k, v in self.kernels.items():
                print("%s : %s" % (k, v))


# The main reason for making this a class is for pretty print
class clContext(object):
    def __init__(self, **kwargs):
        # Unload kwargs
        self.platform_str = kwargs.pop('platform_str', 'Intel Gen OCL Driver')
        self.device_type = kwargs.pop('device_type', 'GPU')
        self.vendor_str = kwargs.pop('vendor_str', 'Intel')
        self.auto_init = kwargs.pop('auto_init', False)
        # Debug
        self.verbose = kwargs.pop('verbose', False)

        # Init internals
        self.context = None
        self.queue = None
        self.platform = None
        self.device = None
        self.kernels = {}

    def __str__(self):
        s = []
        return ''.join(s)

    def get_platform(self):

        platform_list = cl.get_platforms()
        for p in platform_list:
            if p.name == self.platform_str:
                if self.verbose:
                    print("Found platform %s" % p.name)
                return p

        # If we can't find our preferred platform then
        # take the first valid platform we can find
        if len(platform_list) >= 1:
            if self.vebose:
                print("Could not find preferred platform %s" % self.platform_str)
                print("Using alternative platform %s" % platform_list[0].name)
            return platform_list[0]

        return None

    def select_device(self, device_type='GPU'):

        device_list = self.platform.get_devices()
        for d in device_list:
            if cl.device_type.to_string(d.type) == device_type:
                if self.verbose:
                    print("Found device %s" % d.name)
                return d
        # Couldn't get preferred device, take first device in list
        if len(device_list) >= 1:
            if self.verbose:
                print("Unable to find device of type %s" % device_type)
                print("Falling back to device %s" % device_list[0].name)
            return device_list[0]

        return None

    def init_context(self):
        """
        INIT_CONTEXT
        Set up the OpenCL execution context
        """
        self.platform = self.get_platform()
        if self.platform is None:
            raise ValueError('Failed to get a valid platform')

        self.device = self.select_device()      # use the default selection for now
        if self.device is None:
            raise ValueError('Failed to get a valid device')

        # Create the context object
        self.context = cl.Context(dev_type=self.device.type)
        self.queue = cl.CommandQueue(self.context)

    def prog_from_file(self, filename):

        with open(filename, 'r') as fp:
            source = fp.read().replace('\n', '')

        program = clProgram()
        program.build(self.context, source, device=self.device)

