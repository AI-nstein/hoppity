from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import os
import platform
BASEPATH = os.path.dirname(os.path.abspath(__file__))

compile_args = []
link_args = []

if platform.system() != 'Darwin':  # add openmp
    compile_args.append('-fopenmp')
    link_args.append('-lgomp')
    
ext_modules=[CppExtension('extlib', 
                          ['src/extlib.cpp'],
                          extra_compile_args=compile_args,
                          extra_link_args=link_args)]

# build cuda lib
import torch
if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('extlib_cuda',
                                    ['src/extlib_cuda.cpp', 'src/extlib_cuda_kernels.cu']))
 
setup(name='torchext',
      py_modules=['torchext'],
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension}
)
