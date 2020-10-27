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

import json
import numpy as np

from vineyard._C import ObjectMeta
from .utils import build_numpy_buffer, normalize_dtype


def numpy_ndarray_builder(client, value, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Tensor<%s>' % value.dtype.name
    meta['value_type_'] = value.dtype.name
    meta['shape_'] = json.dumps(value.shape)
    meta['partition_index_'] = json.dumps(kw.get('partition_index', []))
    meta['nbytes'] = value.nbytes
    meta.add_member('buffer_', build_numpy_buffer(client, value))
    return client.create_metadata(meta)


def tensor_resolver(obj):
    meta = obj.meta
    value_type = normalize_dtype(meta['value_type_'])
    shape = json.loads(meta['shape_'])
    return np.frombuffer(memoryview(obj.member("buffer_")), dtype=value_type).reshape(shape)


def register_tensor_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(np.ndarray, numpy_ndarray_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', tensor_resolver)
