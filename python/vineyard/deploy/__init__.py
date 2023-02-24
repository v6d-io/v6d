#!/usr/bin/env python
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

import sys

from . import ctl
from . import distributed
from . import kubernetes
from . import local


# generate the `click` commands after import `.ctl`.
def _init():
    ctl._register()  # pylint: disable=no-member
    setattr(sys.modules[__name__], 'vineyardctl', getattr(ctl, 'vineyardctl'))


_init()
del _init
