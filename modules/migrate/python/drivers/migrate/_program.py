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
import subprocess
import sys

import vineyard

base_loc = os.path.join(os.path.dirname(__file__), 'tools')


def _program(name, args):
    if os.name == 'nt':
        name = name + '.exe'
    prog = os.path.join(base_loc, name)

    with vineyard.envvars(
        {
            'LD_LIBRARY_PATH': os.path.dirname(vineyard.__file__),
            'DYLD_LIBRARY_PATH': os.path.dirname(vineyard.__file__),
        }
    ) as env:
        if os.name == 'nt':
            try:
                return subprocess.call([prog] + args, env=env)
            except KeyboardInterrupt:
                return 0
        else:
            return os.execvpe(prog, [prog] + args, env=env)


def vineyard_migrate():
    raise SystemExit(_program('vineyard-migrate', sys.argv[1:]))


def vineyard_migrate_stream():
    raise SystemExit(_program('vineyard-migrate-stream', sys.argv[1:]))


def vineyard_migrate_to_local():
    raise SystemExit(_program('vineyard-migrate-to-local', sys.argv[1:]))
