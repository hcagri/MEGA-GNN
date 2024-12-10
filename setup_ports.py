from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ports_cpp",
        ["ports_cpp.cpp"],
        include_dirs=[
            pybind11.get_include(),  # Include the pybind11 headers
            pybind11.get_include(user=True)  # Include the pybind11 user headers
        ],
        language="c++",
        extra_compile_args=["-fopenmp"],  # Add compiler flag for OpenMP
        extra_link_args=["-fopenmp"],  # Add linker flag for OpenMP
    ),
]

setup(
    name="ports_cpp",
    ext_modules=ext_modules,
)
