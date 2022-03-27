from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name='FastMinv',
      ext_modules=[CUDAExtension('FastMinv', ['M3x3Inv.cpp','Matrix3x3InvKernels.cu'])],
      cmdclass={'build_ext': BuildExtension})