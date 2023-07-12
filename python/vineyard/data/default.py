#! /usr/bin/env python
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

import pickle
from typing import Any
from typing import Optional

from vineyard._C import IPCClient
from vineyard._C import Object
from vineyard._C import ObjectMeta
from vineyard.core.builder import BuilderContext
from vineyard.core.resolver import ResolverContext
from vineyard.data.utils import build_buffer

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle  # pylint: disable=shadow-import


def default_builder(client: IPCClient, value: Any, **kwargs):
    '''Default builder: pickle (version 5), then build a blob object for it.'''
    payload = pickle.dumps(value, protocol=5)
    buffer = build_buffer(client, payload, len(payload))

    meta = ObjectMeta(**kwargs)
    meta['typename'] = 'vineyard::PickleBuffer'
    meta['nbytes'] = len(payload)
    meta['size_'] = len(payload)
    meta.add_member('buffer_', buffer)
    return client.create_metadata(meta)


def default_resolver(obj: Object):
    view = memoryview(obj.member('buffer_'))[0 : int(obj.meta['size_'])]
    return pickle.loads(view, fix_imports=True)


def register_default_types(
    builder_ctx: Optional[BuilderContext] = None,
    resolver_ctx: Optional[ResolverContext] = None,
):
    if builder_ctx is not None:
        builder_ctx.register(object, default_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::PickleBuffer', default_resolver)
