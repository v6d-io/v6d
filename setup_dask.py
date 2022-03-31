#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import textwrap

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel


class bdist_wheel_plat(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        self.root_is_pure = True
        tag = bdist_wheel.get_tag(self)
        self.root_is_pure = False
        return tag


class install_plat(install):
    def finalize_options(self):
        self.install_lib = self.install_platlib
        install.finalize_options(self)


def find_dask_packages(root):
    pkgs = []
    for pkg in find_packages(root):
        if 'contrib.dask' in pkg:
            pkgs.append(pkg)
    return pkgs


with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'),
    encoding='utf-8',
    mode='r',
) as fp:
    long_description = fp.read()

    # Github doesn't respect "align: center", and pypi disables `.. raw`.
    replacement = textwrap.dedent(
        '''
        .. image:: https://v6d.io/_static/vineyard_logo.png
           :target: https://v6d.io
           :align: center
           :alt: vineyard
           :width: 397px

        vineyard: an in-memory immutable data manager
        ---------------------------------------------
        '''
    )
    long_description = replacement + '\n'.join(long_description.split('\n')[8:])

setup(
    name='vineyard-dask',
    author='The vineyard team',
    author_email='developers@v6d.io',
    description='Vineyard integration with Dask',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://v6d.io',
    package_dir={'vineyard.contrib.dask': 'python/vineyard/contrib/dask'},
    packages=find_dask_packages('python'),
    cmdclass={'bdist_wheel': bdist_wheel_plat, "install": install_plat},
    zip_safe=False,
    install_requires=['vineyard', 'dask[complete]', 'click<8.1'],
    platform=['POSIX', 'MacOS'],
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Distributed Computing",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    project_urls={
        'Documentation': 'https://v6d.io',
        'Source': 'https://github.com/v6d-io/v6d',
        'Tracker': 'https://github.com/v6d-io/v6d/issues',
    },
)
