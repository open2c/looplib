#!/usr/bin/env python
import io
import os

import numpy as np

from setuptools import setup
from Cython.Build import cythonize

def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]

install_requires = get_requirements("requirements.txt")

setup(
    name='looplib',
    version='0.1',
    description='A library to simulate loop extrusion on a 1D lattice',
    author='Anton Goloborodko',
    author_email='goloborodko.anton@gmail.com',
    url='https://github.com/golobor/looplib/',
    packages=['looplib'],
    install_requires=install_requires,
    ext_modules = cythonize(['looplib/simlef.pyx',
                             'looplib/simlef_twosided.pyx',
                             'looplib/simlef_onesided.pyx',
                             'looplib/looptools_c.pyx']),
    include_dirs=[np.get_include()]
)
