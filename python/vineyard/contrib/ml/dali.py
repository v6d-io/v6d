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
    from nvidia import dali  # pylint: disable=import-error
    from nvidia.dali import pipeline_def  # pylint: disable=import-error
    from nvidia.dali import types  # pylint: disable=import-error
except ImportError:
    dali = None

from vineyard._C import ObjectMeta
from vineyard.data.utils import build_numpy_buffer
from vineyard.data.utils import from_json
from vineyard.data.utils import normalize_dtype
from vineyard.data.utils import to_json

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


def dali_tensor_resolver(obj, **_kw):
    assert dali is not None, "Nvidia DALI is not available"
    meta = obj.meta
    data_shape = from_json(meta['data_shape_'])
    label_shape = from_json(meta['label_shape_'])
    data_name = meta['data_type_']
    label_name = meta['label_type_']
    data_type = normalize_dtype(data_name, meta.get('value_type_meta_', None))
    label_type = normalize_dtype(label_name, meta.get('value_type_meta_', None))
    data = np.frombuffer(
        memoryview(obj.member('buffer_data_')), dtype=data_type
    ).reshape(data_shape)
    label = np.frombuffer(
        memoryview(obj.member('buffer_label_')), dtype=label_type
    ).reshape(label_shape)
    pipe_out = dali_pipe(  # pylint: disable=unexpected-keyword-arg
        data, label, device_id=device_id, num_threads=num_threads, batch_size=batch_size
    )
    pipe_out.build()  # pylint: disable=no-member
    pipe_output = pipe_out.run()  # pylint: disable=no-member
    return pipe_output


def register_dali_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(
            dali.backend.TensorListCPU, dali_tensor_builder  # noqa: F821
        )

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', dali_tensor_resolver)
