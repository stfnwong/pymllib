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


# The main reason for making this a class is for pretty print
class clContext(object):
    def __init__(self, **kwargs):
        # Unload kwargs
        self.platform_str = kwargs.pop('platform_str', 'Intel Gen OCL Driver')
        self.device_type = kwargs.pop('device_type', 'GPU')
        self.vendor_str = kwargs.pop('vendor_str', 'Intel')

        # Init internals
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        self.platform = self._get_platform()

        if self.platform is None:
            raise ValueError("Failed to get a valid platform")

        #self.platform_id = None

    def __str__(self):
        s = []
        return ''.join(s)

    def _get_platform(self):
        platform_list = cl.get_platforms()

        for p in platform_list:
            if p.name == self.platform_str:
                return p    # use this platform

        # There is only one platform
        if len(platform_list) == 1:
            return platform_list[0]

        return None
