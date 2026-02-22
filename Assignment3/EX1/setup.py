from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("gauss_seidel.pyx", annotate=True),
    include_dirs=[np.get_include()]
)