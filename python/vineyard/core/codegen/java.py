#! /usr/bin/env python
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
import pathlib

from sphinx import subprocess

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
    root_directory,
    source,
    target,
    package,
    system_includes=None,
    includes=None,
    extra_flags=None,
    build_directory=None,
    verbose=None,
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

    # known unsupported classes
    known_unsupported = [
        '--exclude',
        'std.map',
        '--exclude',
        'std.set',
        '--exclude',
        'std.unordered_map',
        '--exclude',
        'std.vector',
        '--exclude',
        'nonstd.sv_lite.basic_string_view',
        '--exclude',
        '.*::list_type',
        '--exclude',
        'arrow.LargeStringType' '--exclude',
        'arrow.LargeBinaryType' '--exclude',
        'arrow.Buffer.Copy',
    ]

    # execute
    cmd = [
        'java',
        '-jar',
        find_ffi_binding_generator(),
        '--output-directory',
        package,
        '--debug',
        *known_unsupported,
        '--',
        *flags,
        source,
    ]

    proc = None
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
    if proc is not None and proc.returncode != 0:
        print(proc.stdout)

    # touch the file to ensure the dependencies
    os.makedirs(os.path.dirname(target), exist_ok=True)
    pathlib.Path(target).touch(exist_ok=True)
