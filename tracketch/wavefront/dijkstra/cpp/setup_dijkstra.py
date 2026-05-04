"""
Setup script to build the C++ Dijkstra extension.

Build with:
    pip install pybind11
    python setup_dijkstra.py build_ext --inplace

Or install:
    pip install .
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "dijkstra_cpp",
        ["dijkstra_cpp.cpp"],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="dijkstra_cpp",
    version="0.1.0",
    author="Tracketch project",
    description="Fast C++ Dijkstra for 2D non-uniform grids",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
