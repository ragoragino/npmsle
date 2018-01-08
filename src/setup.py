import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)

sourcefiles = [r"npsmle_def.pyx", r"main.cpp"]

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("npsmle", sourcefiles, language="c++",
                             include_dirs=[".", np.get_include(), os.path.dirname(cur_dir) + r'/include'],
							 library_dirs = [os.path.dirname(cur_dir) + r'/libs'], libraries=['libnlopt-0'],
							 extra_compile_args=['/O2'])])