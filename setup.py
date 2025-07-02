#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import os

__name__ = 'cloe'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           __name__, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

version = release_info['__version__']

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Function to automatically find scripts.
def find_scripts():
    sdir = './scripts'
    return [os.path.join(sdir, val) for val in os.listdir(sdir) if
            val.endswith('.py') and '__init__' not in val]

# pybind11 modules
try:
    cflags = subprocess.check_output(['gsl-config', '--cflags'], universal_newlines=True).split()
    lflags = subprocess.check_output(['gsl-config', '--libs'], universal_newlines=True).split()
except OSError:
    raise Exception("Error: must have GSL installed and gsl-config working")

extra_compile_args = [os.path.expandvars(flag) for flag in cflags]
extra_link_args = [os.path.expandvars(flag) for flag in lflags]

ext_modules = [
    Pybind11Extension('integration_routines',
        ['cloe/clusters_of_galaxies/cpp_modules/integration_routines.cpp'],
        define_macros = [('VERSION_INFO', version)],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args
        ),
]

# Setup
setup(
    name=__name__,
    author=release_info['__author__'],
    author_email=release_info['__email__'],
    version=version,
    url=release_info['__url__'],
    packages=find_packages(),
    scripts=find_scripts(),
    include_package_data=True,
    install_requires=release_info['__requires__'],
    license=release_info['__license__'],
    description=release_info['__about__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=release_info['__setup_requires__'],
    tests_require=release_info['__tests_require__'],
    ext_modules=ext_modules
)
