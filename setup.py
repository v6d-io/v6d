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

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


class CopyCMakeExtension(Extension):
    def __init__(self, name):
        super(CopyCMakeExtension, self).__init__(name, sources=[])


class CopyCMakeBin(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_py = self.get_finalized_command('build_py')
        package_dir = os.path.abspath(build_py.get_package_dir(''))
        bin_path = os.path.join(package_dir, self.get_ext_filename(ext.name))
        target_path = self.get_ext_fullpath(ext.name)
        self.copy_file(bin_path, target_path)


class bdist_wheel_injected(bdist_wheel):
    def finalize_options(self):
        super(bdist_wheel_injected, self).finalize_options()
        self.root_is_pure = False


class BinDistribution(Distribution):
    ''' Always forces a binary package with platform name.
    '''
    def has_ext_modules(self):
        return True


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8', mode='r') as fp:
    long_description = fp.read()

    # Github doesn't respect "align: center", and pypi disables `.. raw`.
    replacement = textwrap.dedent('''
        .. image:: https://v6d.io/_static/vineyard_logo.png
           :target: https://v6d.io
           :align: center
           :alt: vineyard
           :width: 397px

        vineyard: an in-memory immutable data manager
        ---------------------------------------------
        ''')
    long_description = replacement + '\n'.join(long_description.split('\n')[8:])

setup(
    name='vineyard',
    author='The vineyard team',
    author_email='developers@v6d.io',
    description='An in-memory immutable data manager',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://v6d.io',
    package_dir={'': 'python'},
    packages=find_packages('python'),
    package_data={
        'vineyard': [
            "vineyardd",
            "**/*.yaml",
            "**/*.yaml.tpl",
            "**/**/*.sh",
        ],
    },
    ext_modules=[
        CopyCMakeExtension('vineyard._C'),
    ],
    cmdclass={
        'build_ext': CopyCMakeBin,
        'bdist_wheel': bdist_wheel_injected,
    },
    distclass=BinDistribution,
    zip_safe=False,
    entry_points={
        'cli': ['vineyard-codegen=vineyard.cli:main'],
        'console_scripts': ['vineyard-codegen=vineyard.core.codegen:main'],
    },
    setup_requires=[
        'libclang',
        'parsec',
        'setuptools',
        'wheel',
    ],
    install_requires=[
        'numpy',
        'pandas<1.0.0; python_version<"3.6"',
        'pandas<1.2.0; python_version<"3.7"',
        'pandas>=1.0.0,<1.3.0; python_version>="3.7"',
        'pickle5; python_version<="3.7"',
        'pyarrow',
        'setuptools',
        'shared-memory38; python_version<="3.7"',
        'sortedcontainers',
    ],
    extras_require={
        'dev': [
            'breathe',
            'libclang',
            'parsec',
            'pytest',
            'pytest-benchmark',
            'pytest-datafiles',
            'sphinx>=3.0.2',
            'sphinx_rtd_theme',
            'docutils==0.16',
        ],
        "kubernetes": [
            "kubernetes",
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
