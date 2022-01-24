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
import subprocess
import sys
import textwrap
from distutils.cmd import Command

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

repo_root = os.path.dirname(os.path.abspath(__file__))


class CopyCMakeExtension(Extension):
    def __init__(self, name):
        super(CopyCMakeExtension, self).__init__(name, sources=[])


class build_ext_with_precompiled(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_py = self.get_finalized_command('build_py')
        package_dir = os.path.abspath(build_py.get_package_dir(''))
        bin_path = os.path.join(package_dir, self.get_ext_filename(ext.name))
        target_path = self.get_ext_fullpath(ext.name)
        self.copy_file(bin_path, target_path)


class build_py_with_dependencies(build_py):
    def _get_data_files(self):
        """Add custom out-of-tree package data files."""
        rs = super()._get_data_files()

        package = 'vineyard'
        src_dir = os.path.abspath(self.get_package_dir(package))
        build_dir = os.path.join(self.build_lib, package)
        if sys.platform == 'linux':
            filenames = ['libvineyard_internal_registry.so']
        elif sys.platform == 'darwin':
            filenames = ['libvineyard_internal_registry.dylib']
        else:
            raise RuntimeError('Unsupported platform: %s' % sys.platform)

        rs.append((package, src_dir, build_dir, filenames))
        return rs


class bdist_wheel_as_nonpure(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


class BinDistribution(Distribution):
    """Always forces a binary package with platform name."""

    def has_ext_modules(self):
        return True


class FormatAndLint(Command):
    description = 'format and lint code'
    user_options = []

    user_options = [('inplace=', 'i', 'Run code formatter and linter inplace')]

    def initialize_options(self):
        self.inplace = False

    def finalize_options(self):
        if self.inplace or self.inplace == 'True' or self.inplace == 'true':
            self.inplace = True
        else:
            self.inplace = False

    def run(self):
        targets = [
            'python/',
            'modules/',
            'setup.py',
            'setup_airflow.py',
            'setup_dask.py',
            'setup_ml.py',
            'setup_ray.py',
            'test/runner.py',
        ]

        if self.inplace:
            subprocess.check_call(
                [sys.executable, '-m', 'isort'] + targets, cwd=repo_root
            )
            subprocess.check_call(
                [sys.executable, '-m', 'black'] + targets, cwd=repo_root
            )
            subprocess.check_call(
                [sys.executable, '-m', 'flake8'] + targets, cwd=repo_root
            )
        else:
            subprocess.check_call(
                [sys.executable, '-m', 'isort', '--check', '--diff'] + targets,
                cwd=repo_root,
            )
            subprocess.check_call(
                [sys.executable, '-m', 'black', '--check', '--diff'] + targets,
                cwd=repo_root,
            )
            subprocess.check_call(
                [sys.executable, '-m', 'flake8'] + targets, cwd=repo_root
            )


def find_core_packages(root):
    pkgs = []
    for pkg in find_packages(root):
        if 'contrib' not in pkg or pkg.endswith('.contrib'):
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
        """
        .. image:: https://v6d.io/_static/vineyard_logo.png
           :target: https://v6d.io
           :align: center
           :alt: vineyard
           :width: 397px

        vineyard: an in-memory immutable data manager
        ---------------------------------------------
        """
    )
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
    packages=find_core_packages('python'),
    package_data={
        'vineyard': [
            'vineyardd',
            '**/*.yaml',
            '**/*.yaml.tpl',
            '**/**/*.sh',
        ],
    },
    ext_modules=[
        CopyCMakeExtension('vineyard._C'),
    ],
    cmdclass={
        'build_ext': build_ext_with_precompiled,
        'build_py': build_py_with_dependencies,
        'bdist_wheel': bdist_wheel_as_nonpure,
        'lint': FormatAndLint,
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
        'argcomplete',
        'etcd-distro',
        'numpy>=0.18.5',
        'pandas<1.0.0; python_version<"3.6"',
        'pandas<1.2.0; python_version<"3.7"',
        'pandas>=1.0.0; python_version>="3.7"',
        'pickle5; python_version<="3.7"',
        'psutil',
        'pyarrow',
        'setuptools',
        'shared-memory38; python_version<="3.7"',
        'sortedcontainers',
        'treelib',
    ],
    extras_require={
        'dev': [
            'black',
            'breathe',
            'docutils==0.16',
            'flake8',
            'isort',
            'libclang',
            'nbsphinx',
            'parsec',
            'pygments>=2.4.1',
            'pytest',
            'pytest-benchmark',
            'pytest-datafiles',
            'sphinx>=3.0.2',
            'sphinx_rtd_theme',
        ],
        'kubernetes': [
            'kubernetes',
        ],
    },
    platform=["POSIX", "MacOS"],
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
        "Documentation": "https://v6d.io",
        "Source": "https://github.com/v6d-io/v6d",
        "Tracker": "https://github.com/v6d-io/v6d/issues",
    },
)
