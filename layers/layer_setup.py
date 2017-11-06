from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

"""
Run python3 layer_setup.py build_ext --inplace
"""

extensions = [
    Extension('im2col_cython', ['layers/im2col_cython.pyx'],
              include_dirs = [numpy.get_include()]
              ),
]

setup(
    ext_modules = cythonize(extensions),
)
