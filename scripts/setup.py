from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize("utility/RoundaboutChecker.pyx"))
setup(ext_modules=cythonize("utility/RoundaboutEntering.pyx"))
# setup(ext_modules = cythonize("utility/transform.pyx"))
