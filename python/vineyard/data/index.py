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

import json
import pandas as pd

from vineyard._C import ObjectMeta

from .utils import from_json, to_json, normalize_dtype


def pandas_index_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Index'
    meta['name'] = to_json(value.name)
    meta['value_type_'] = value.dtype.name
    meta.add_member('value_', builder.run(client, value.to_numpy(), **kw))
    return client.create_metadata(meta)


def pandas_index_resolver(obj, resolver):
    meta = obj.meta
    value_type = normalize_dtype(meta['value_type_'])
    name = from_json(meta['name'])
    value = resolver.run(obj.member('value_'))
    return pd.Index(value, dtype=value_type, name=name)


def register_index_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(pd.Index, pandas_index_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Index', pandas_index_resolver)
