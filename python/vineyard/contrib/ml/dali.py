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

import numpy as np

try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.types as types
except ImportError:
    dali = None

from vineyard._C import ObjectMeta
from vineyard.data.utils import from_json, to_json, build_numpy_buffer, normalize_dtype

num_gpus = 1
device_id = 0
batch_size = 2
num_threads = 4

if dali is not None:

    @pipeline_def
    def dali_pipe(data, label):
        fdata = types.Constant(data)
        flabel = types.Constant(label)
        return fdata, flabel


def dali_tensor_builder(client, value, **kw):
    assert dali is not None, "Nvidia DALI is not available"
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Tensor'
    meta['partition_index_'] = to_json(kw.get('partition_index', []))
    data = np.array(value[0])
    label = np.array(value[1])
    meta.add_member('buffer_data_', build_numpy_buffer(client, data))
    meta.add_member('buffer_label_', build_numpy_buffer(client, label))
    meta['data_shape_'] = to_json(data.shape)
    meta['label_shape_'] = to_json(label.shape)
    meta['data_type_'] = data.dtype.name
    meta['label_type_'] = label.dtype.name
    meta['data_type_meta_'] = data.dtype.str
    meta['label_type_meta_'] = label.dtype.str
    return client.create_metadata(meta)


def dali_tensor_resolver(obj, **kw):
    assert dali is not None, "Nvidia DALI is not available"
    meta = obj.meta
    data_shape = from_json(meta['data_shape_'])
    label_shape = from_json(meta['label_shape_'])
    data_name = meta['data_type_']
    label_name = meta['label_type_']
    data_type = normalize_dtype(data_name, meta.get('value_type_meta_', None))
    label_type = normalize_dtype(label_name, meta.get('value_type_meta_', None))
    data = np.frombuffer(memoryview(obj.member('buffer_data_')), dtype=data_type).reshape(data_shape)
    label = np.frombuffer(memoryview(obj.member('buffer_label_')), dtype=label_type).reshape(label_shape)
    pipe_out = dali_pipe(data, label, device_id=device_id, num_threads=num_threads, batch_size=batch_size)
    pipe_out.build()
    pipe_output = pipe_out.run()
    return pipe_output


def register_dali_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(nvidia.dali.backend.TensorListCPU, dali_tensor_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', dali_tensor_resolver)
