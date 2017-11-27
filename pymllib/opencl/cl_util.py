"""
CL_UTIL

Various OpenCL utilities
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyopencl as cl


def cl_load_kernel(filename):
    with open(filename, 'r') as fp:
        kernel_source = fp.readlines()

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
    def __init__(self):
        self.source = ''
        self.filename = ''
        self.prg = None

    def __str__(self):
        s = []
        s.append('Source file : %s\n' % self.filename)
        # TODO : list the names of all kernel functons that appear
        # in the compiled source

        return ''.join(s)

    def __repr__(self):
        return self.filename

    def load_source(self, filename):
        with open(filename, 'r') as fp:
            self.source = fp.read()
        self.filename = filename

    def build(self, ctx):
        self.prg = cl.Program(ctx, self.source)

# The main reason for making this a class is for pretty print
class clContext(object):
    def __init__(self, **kwargs):
        # Unload kwargs
        self.platform_str = kwargs.pop('platform_str', 'Intel Gen OCL Driver')
        self.device_type = kwargs.pop('device_type', 'GPU')
        self.vendor_str = kwargs.pop('vendor_str', 'Intel')
        # Debug
        self.vebose = kwargs.pop('verbose', False)

        # Init internals
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        self.platform = self._get_platform()
        self.device = None
        self.kernels = []

        if self.platform is None:
            raise ValueError("Failed to get a valid platform")

    def __str__(self):
        s = []
        return ''.join(s)

    def _get_platform(self):
        platform_list = cl.get_platforms()

        for p in platform_list:
            if p.name == self.platform_str:
                return p    # use this platform

        # If we can't find our preferred platform then
        # take the first valid platform we can find
        if len(platform_list) >= 1:
            return platform_list[0]

        return None

    def get_device_list(self):
        return self.platform.get_devices()

    def select_device(self, device_type='GPU'):

        device_list = self.platform.get_devices()

        for d in device_list:
            if cl.device_type.to_string(d.name) == device_type:
                self.device = d
                return
        # Couldn't get preferred device, take first device in list
        self.device = device_list[0]
        return

    def load_kernel(self, filename):
        kernel = clKernel()
        kernel.load_source(filename)
        self.kernels.append(kernel)

    def build_program(self):
        # TODO : How to get compiler error messages?
        for k in self.kernels:
            k.build(self.context)

