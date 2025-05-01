from setuptools import setup, Extension
import numpy as np

fe_interpolator_ext = Extension(
    'qdsim.fe_interpolator_ext',
    sources=['qdsim/fe_interpolator_ext.cpp'],
    include_dirs=[np.get_include(), '/usr/include/eigen3'],
    extra_compile_args=['-std=c++17'],
)

mesh_adapter = Extension(
    'qdsim.mesh_adapter',
    sources=['qdsim/mesh_adapter.cpp'],
    include_dirs=[np.get_include(), '/usr/include/eigen3'],
    extra_compile_args=['-std=c++17'],
)

setup(
    name='qdsim',
    version='0.1',
    description='Quantum Dot Simulator',
    packages=['qdsim'],
    ext_modules=[fe_interpolator_ext, mesh_adapter],
)
