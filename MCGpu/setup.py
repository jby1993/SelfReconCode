from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name='MCGpu',
      ext_modules=[CUDAExtension('MCGpu', ['MCGpu.cpp','CudaKernels.cu'])],
      cmdclass={'build_ext': BuildExtension})