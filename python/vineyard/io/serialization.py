#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
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

import logging
import traceback
from urllib.parse import urlparse

import vineyard
from ..core.driver import registerize

logger = logging.getLogger('vineyard')


@registerize
def serialize(path, object_id, *args, **kwargs):
    parsed = urlparse(path)
    if not parsed.scheme:
        path = 'file://' + path
    obj_type = kwargs.pop('type', 'global')
    if serialize.__factory and serialize.__factory[obj_type]:
        proc_kwargs = kwargs.copy()
        serializer = serialize.__factory[obj_type][0]
        serializer(path, object_id, proc_kwargs.pop('vineyard_ipc_socket'), *args, **proc_kwargs)
        return
    raise RuntimeError('Unable to find a proper seraialization driver for %s' % path)


@registerize
def deserialize(path, *args, **kwargs):
    parsed = urlparse(path)
    if not parsed.scheme:
        path = 'file://' + path
    obj_type = kwargs.pop('type', 'global')
    if deserialize.__factory and deserialize.__factory[obj_type]:
        proc_kwargs = kwargs.copy()
        deserializer = deserialize.__factory[obj_type][0]
        return deserializer(path, proc_kwargs.pop('vineyard_ipc_socket'), *args, **proc_kwargs)
    raise RuntimeError('Unable to find a proper deseraialization driver for %s' % path)


__all__ = ['serialize', 'deserialize']
