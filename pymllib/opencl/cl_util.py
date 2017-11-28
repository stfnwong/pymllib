"""
CL_UTIL

Various OpenCL utilities
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyopencl as cl

# Debug
#from pudb import set_trace; set_trace()


def cl_load_kernel(filename):
    with open(filename, 'r') as fp:
        kernel_source = fp.read().replace('\n', '')

    return kernel_source

def cl_select_platform(pname, verbose=False):
    # Try to get our preferred platform
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


class clProgram(object):
    """
    CLPROGRAM
    Builds a program from a source file and returns a dictionary of
    kernel references. Each key in the dictionary is the name of a
    kernel in the source file.
    """
    def __init__(self, verbose=False):
        self.prog = None
        self.kernels = {}
        self.verbose = verbose

    def __str__(self):
        s =[]
        if self.kernels:
            s.append('Program contains:\n')
            for k, v in self.kernels.items():
                s.append('\t%s : %s' % (k, v))

        return ''.join(s)

    def __repr__(self):
        return self.__str__()

    def build(self, ctx, source, device, options=[]):

        self.prog = cl.Program(ctx, source).build(devices=[device])
        # Print build log to console in verbose mode
        if self.verbose:
            build_log = self.prog.get_build_info(device, cl.program_build_info.LOG)
            if len(build_log) > 0:
                if self.verbose:
                    print("Build log:")
                    print('%s\n' % build_log)

        if self.prog.num_kernels < 1:
            err_str = "No kernels were built for program %s" % self.prog
            raise ValueError(err_str)

        # Create references to the kernels in this program
        kernel_list = self.prog.all_kernels()
        for k in kernel_list:
            self.kernels[k.function_name] = k

        return self.kernels

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
        if self.platform is not None:
            s.append('Platform Name   : %s\n' % self.platform.name)
            s.append('Platform Vendor : %s\n' % self.platform.vendor)
        if self.device is not None:
            s.append('Device Name     : %s\n' % self.device.name)
            s.append('Device Vendor   : %s\n' % self.device.vendor)
        if self.kernels:
            print('Kernels :\n')
            for k, v in self.kernels.items():
                s.append('%s : %s' % (k, v))

        return ''.join(s)

    def _update_program(self, program):
        # TODO : check for attribute clash?
        self.kernels.update(program.kernels)

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
            if self.verbose:
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

    def load_source(self, filename):

        with open(filename, 'r') as fp:
            source = fp.read().rstrip('\n')

        prog = clProgram(verbose=self.verbose)
        kernels = prog.build(self.context, source, self.device)
        if not kernels:
            return None
        self.kernels.update(kernels)

        return len(kernels.keys())
