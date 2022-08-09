#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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
import pathlib
import subprocess
import sys

from .parsing import generate_parsing_flags


def find_java_command():
    javahome = os.environ.get('JAVA_HOME', None)
    if javahome is None:
        return 'java'
    else:
        return os.path.join(javahome, 'bin', 'java')


def find_ffi_binding_generator():
    generator = os.environ.get('FFI_BINDING_GENERATOR', None)
    if generator is None:
        mvnrepo = os.environ.get('MVN_REPOSITORY', None)
        if mvnrepo is None:
            m2 = os.environ.get('M2', None)
            if m2 is None:
                m2 = os.path.expanduser('~/.m2')
            mvnrepo = os.path.join(m2, 'repository')
        generator = os.path.join(
            mvnrepo,
            'com/alibaba/fastffi/binding-generator/',
            '0.1',
            'binding-generator-0.1-jar-with-dependencies.jar',
        )
    return generator


def codegen(
    root_directory,  # pylint: disable=unused-argument
    source,
    target,
    package,
    system_includes=None,
    includes=None,
    extra_flags=None,
    build_directory=None,
    verbose=None,  # pylint: disable=unused-argument
    package_name=None,
    ffilibrary_name=None,
    excludes=None,
    forwards=None,
):
    if os.path.exists(target):
        try:
            os.unlink(target)
        except OSError:
            pass
    os.makedirs(package, exist_ok=True)

    parsing_flags = generate_parsing_flags(
        source,
        system_includes=system_includes,
        includes=includes,
        extra_flags=extra_flags,
        build_directory=build_directory,
    )

    flags = [
        # -p build_directory: required by clang tooling to discover
        # compilation database.
        #
        # see also: CommonOptionsParser.cpp in llvm-project
        '-p',
        build_directory,
        *['--extra-arg-before=%s' % flag for flag in parsing_flags],
    ]

    # execute
    cmd = [
        'java',
        '-jar',
        find_ffi_binding_generator(),
        '--output-directory',
        package,
    ]
    if package_name:
        cmd.extend(['--root-package', package_name])
    if ffilibrary_name:
        cmd.extend(['--ffilibrary-name', ffilibrary_name])
    if excludes:
        cmd.extend(['--excludes-file', excludes])
    if forwards:
        cmd.extend(['--forward-headers-file', forwards])

    cmd.append('--')
    cmd.extend(flags)
    cmd.append(source)

    proc = None
    try:
        proc = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(e, file=sys.stderr)
        sys.exit(-1)
    else:
        # touch the file to ensure the dependencies
        os.makedirs(os.path.dirname(target), exist_ok=True)
        pathlib.Path(target).touch(exist_ok=True)
    finally:
        if proc is not None and proc.returncode != 0:
            sys.exit(-1)
