from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

"""
Run python3 setup.py build_ext --inplace
"""

extensions = [
    Extension('pymllib.layers.im2col_cython', ['pymllib/layers/im2col_cython.pyx'],
              include_dirs = [numpy.get_include()]
              ),
]

install_requires=['Cython', 'h5py']

setup(
    packages=['pymllib', 'tests'],
    ext_modules = cythonize(extensions),
)
# TODO : These are just lifted from the ARL repo.... but I wonder
# how can these variables be used here....
#    test_suite='tests',
#    tests_require=['pytest'],
