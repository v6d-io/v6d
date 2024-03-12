#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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
from configparser import ConfigParser
from distutils.cmd import Command
from distutils.util import strtobool
from typing import List

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

repo_root = os.path.dirname(os.path.abspath(__file__))

try:
    cf = ConfigParser()
    cf.read(os.path.join(repo_root, 'setup.cfg'))
    __version__ = cf['metadata']['version']
    vineyard_bdist = 'vineyard-bdist==%s' % __version__
except:  # noqa: E722
    __version__ = None
    vineyard_bdist = 'vineyard-bdist'


def value_to_bool(val):
    if isinstance(val, str):
        return strtobool(val)
    return val


class CopyCMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


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

    user_options = [
        ('inplace=', 'i', 'Run code formatter and linter inplace'),
        ('pylint=', None, 'Run the pylint checker'),
        ('flake8=', None, 'Run the flake8 checker'),
    ]

    def initialize_options(self):
        self.inplace = False
        self.pylint = False
        self.flake8 = True

    def finalize_options(self):
        self.inplace = value_to_bool(self.inplace)
        self.pylint = value_to_bool(self.pylint)
        self.flake8 = value_to_bool(self.flake8)

    linter_targets = [
        'python/',
        'setup.py',
        'setup_airflow.py',
        'setup_dask.py',
        'setup_io.py',
        'setup_ml.py',
        'setup_ray.py',
        'test/runner.py',
        'modules/fuse/test',
    ]

    def extend_cmd_as_import(self, cmd) -> List[str]:
        if not isinstance(cmd, (list, tuple)):
            cmd = [cmd]
        return [sys.executable, '-m'] + cmd

    def linter(self, cmd):
        cmd = self.extend_cmd_as_import(cmd)
        subprocess.check_call(cmd + self.linter_targets, cwd=repo_root)

    def linter_pylint(self, cmd):
        cmd = self.extend_cmd_as_import(cmd)
        # lint misc
        subprocess.check_call(cmd + self.linter_targets[1:], cwd=repo_root)
        # lint main package
        subprocess.check_call(cmd + ['vineyard'], cwd=os.path.join(repo_root, 'python'))

    def run(self):
        if self.inplace:
            self.linter(['isort'])
            self.linter(['black'])
            self.linter(['flake8'])
        else:
            self.linter(['isort', '--check', '--diff'])
            self.linter(['black', '--check', '--diff'])
            self.linter(['flake8'])
        if self.pylint:
            self.linter_pylint(
                ['pylint', '--rcfile=%s' % os.path.join(repo_root, '.pylintrc')]
            )
        if self.flake8:
            self.linter(['flake8'])


def find_core_packages(root):
    pkgs = []
    for pkg in find_packages(root):
        if 'contrib' in pkg and not pkg.endswith('.contrib'):
            continue
        if 'drivers' in pkg and not pkg.endswith('.drivers'):
            continue
        if 'llm' in pkg:
            continue
        pkgs.append(pkg)
    return pkgs


def load_requirements_txt(kind=""):
    requirements = []
    with open(
        os.path.join(repo_root, "requirements%s.txt" % kind), "r", encoding="utf-8"
    ) as fp:
        for req in fp.read().splitlines():
            if '#' in req:
                req = req.split('#')[0]
            req = req.strip()
            if req:
                requirements.append(req)
    return requirements


def package_data():
    artifacts = [
        '*.pyi',
        'deploy/*.yaml',
        'deploy/*.yaml.tpl',
    ]
    return artifacts


with open(
    os.path.join(repo_root, 'README.rst'),
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
        'vineyard': package_data(),
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
        'vineyardctl': ['vineyardctl=vineyard.deploy.ctl:_main'],
        'console_scripts': ['vineyard-codegen=vineyard.core.codegen:main'],
    },
    setup_requires=load_requirements_txt("-setup"),
    install_requires=load_requirements_txt() + [vineyard_bdist],
    extras_require={
        'dev': load_requirements_txt("-dev"),
        'extra': load_requirements_txt("-extra"),
        'kubernetes': load_requirements_txt("-kubernetes"),
    },
    platforms=["POSIX", "MacOS"],
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "Documentation": "https://v6d.io",
        "Source": "https://github.com/v6d-io/v6d",
        "Tracker": "https://github.com/v6d-io/v6d/issues",
    },
)
