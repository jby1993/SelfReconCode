from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='interplate',
      ext_modules=[CUDAExtension('interp2x_boundary2d', ['interp2x_boundary2d.cpp','interp2x_boundary2d_kernel.cu']),
                CUDAExtension('interp2x_boundary3d', ['interp2x_boundary3d.cpp','interp2x_boundary3d_kernel.cu']),
                CUDAExtension('GridSamplerMine', ['GridSamplerMine.cpp','GridSamplerMineKernel.cu']),],
      cmdclass={'build_ext': BuildExtension})

# setup(name='interplate',
#       ext_modules=[CUDAExtension('GridSamplerMine', ['GridSamplerMine.cpp','GridSamplerMineKernel.cu'])],
#       cmdclass={'build_ext': BuildExtension})