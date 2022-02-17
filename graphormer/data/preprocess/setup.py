from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graphormer_preprocess_cuda',
    ext_modules=[
        CUDAExtension('graphormer_preprocess_cuda', [
            'graphormer_preprocess_cuda.cpp',
            'graphormer_preprocess_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
