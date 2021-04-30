"""
    Code taken from Hyperbolic Hierarchical Clustering (HypHC) by Chami et al.
    for more details visit https://github.com/HazyResearch/HypHC
"""
# from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("unionfind.pyx", annotate=True, language_level="3"),
    include_dirs=[numpy.get_include()],
)
