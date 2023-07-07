# 这个文件将 python 编译为 c，运行速度有很大提升
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules = cythonize("scripts/ImgProcess.pyx"))
# setup(ext_modules = cythonize("scripts/Main.pyx"))
# setup(ext_modules = cythonize("scripts/transform.pyx"))
# setup(ext_modules = cythonize("scripts/ImgWindow.pyx"))
