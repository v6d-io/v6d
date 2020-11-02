#!/usr/env/env python3
# -*- coding: utf-8 -*-

import os

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


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'),
          encoding='utf-8',
          mode='r') as fp:
    long_description = fp.read()


setup(
    name='vineyard',
    version='0.0.1',
    author='7br',
    author_email='7br@alibaba-inc.com',
    description='An in-memory immutable data manager',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    package_dir={'': 'python'},
    packages=find_packages('python'),
    package_data={
        'vineyard': [
            '*/**/*.sh',
            '*/**/*.yaml',
            '*/**/*.bin',
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
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow',
        'sortedcontainers',
        'setuptools',
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
        ],
    },
)
