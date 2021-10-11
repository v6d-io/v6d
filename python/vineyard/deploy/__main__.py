#!/usr/bin/env python
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

from .utils import find_vineyardd_path


def deploy_vineyardd(args):
    try:
        vineyardd = find_vineyardd_path()
        if vineyardd is None:
            raise RuntimeError("Unable to vineyardd")
        if os.name == 'nt':
            return subprocess.call([vineyardd] + args)
        else:
            return os.execvp(vineyardd, [vineyardd] + args)
    except KeyboardInterrupt:
        return 0


def main():
    raise SystemExit(deploy_vineyardd(sys.argv[1:]))


if __name__ == '__main__':
    main()
