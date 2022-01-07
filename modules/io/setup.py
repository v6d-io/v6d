#!/usr/env/env python3
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


repo_root = os.path.dirname(os.path.abspath(__file__))


def find_vineyard_io_packages():
    packages = []

    for pkg in find_packages("python/drivers"):
        packages.append('vineyard.drivers.%s' % pkg)

    return packages


def resolve_vineyard_io_package_dir():
    package_dir = {
        "vineyard.drivers": "python/drivers",
    }
    return package_dir


with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'),
    encoding='utf-8',
    mode='r',
) as fp:
    long_description = fp.read()

setup(
    name='vineyard-io',
    author='The vineyard team',
    author_email='developers@alibaba-inc.com',
    description='IO drivers for vineyard',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://v6d.io',
    packages=find_vineyard_io_packages(),
    package_dir=resolve_vineyard_io_package_dir(),
    package_data={
        'vineyard.drivers.io': ['*.sh'],
    },
    include_package_data=True,
    zip_safe=False,
    cmdclass={'bdist_wheel': bdist_wheel_plat, "install": install_plat},
    entry_points={
        'console_scripts': [
            'vineyard_deserializer=' 'vineyard.drivers.io.adaptors.deserializer:main',
            'vineyard_dump_dataframe='
            'vineyard.drivers.io.adaptors.dump_dataframe:main',
            'vineyard_parse_bytes_to_dataframe='
            'vineyard.drivers.io.adaptors.parse_bytes_to_dataframe:main',
            'vineyard_parse_dataframe_to_bytes='
            'vineyard.drivers.io.adaptors.parse_dataframe_to_bytes:main',
            'vineyard_read_bytes=' 'vineyard.drivers.io.adaptors.read_bytes:main',
            'vineyard_read_bytes_collection='
            'vineyard.drivers.io.adaptors.read_bytes_collection:main',
            'vineyard_read_orc=' 'vineyard.drivers.io.adaptors.read_orc:main',
            'vineyard_read_vineyard_dataframe='
            'vineyard.drivers.io.adaptors.read_vineyard_dataframe:main',
            'vineyard_serializer=' 'vineyard.drivers.io.adaptors.serializer:main',
            'vineyard_write_bytes=' 'vineyard.drivers.io.adaptors.write_bytes:main',
            'vineyard_write_bytes_collection='
            'vineyard.drivers.io.adaptors.write_bytes_collection:main',
            'vineyard_write_orc=' 'vineyard.drivers.io.adaptors.write_orc:main',
            'vineyard_write_vineyard_dataframe='
            'vineyard.drivers.io.adaptors.write_vineyard_dataframe:main',
        ],
    },
    setup_requires=[
        'setuptools',
        'wheel',
    ],
    extras_require={
        'dev': [
            'pytest',
        ],
    },
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
