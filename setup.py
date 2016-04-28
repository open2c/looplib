#!/usr/bin/env python
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='looplib',
    version='0.1',
    description='A library to simulate loop extrusion on a 1D lattice',
    author='Anton Goloborodko',
    author_email='goloborodko.anton@gmail.com',
    url='https://github.com/golobor/looplib/',
    packages=['looplib'],
    install_requires=['numpy', 'matplotlib'],
    ext_modules = cythonize(['looplib/simlef.pyx',
                             'looplib/simlef_twosided.pyx',
                             'looplib/looptools_c.pyx']),
    include_dirs=[np.get_include()]
)
