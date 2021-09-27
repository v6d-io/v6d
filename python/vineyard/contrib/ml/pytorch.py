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
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from vineyard._C import ObjectMeta
from vineyard.core.resolver import resolver_context, default_resolver_context
from vineyard.data.utils import from_json, to_json, build_numpy_buffer, normalize_dtype


def torch_tensor_builder(client, value, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Tensor'
    meta['partition_index_'] = to_json(kw.get('partition_index', []))
    data = value
    data = DataLoader(data, batch_size=len(value))
    for x, y in data:
        meta.add_member('buffer_data_', build_numpy_buffer(client, x.numpy()))
        meta.add_member('buffer_label_', build_numpy_buffer(client, y.numpy()))
        meta['data_shape_'] = to_json(x.numpy().shape)
        meta['label_shape_'] = to_json(y.numpy().shape)
        meta['data_type_'] = x.numpy().dtype.name
        meta['label_type_'] = y.numpy().dtype.name
        meta['data_type_meta_'] = x.numpy().dtype.str
        meta['label_type_meta_'] = y.numpy().dtype.str
    return client.create_metadata(meta)


def torch_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    cols = kw.get('cols')
    label = kw.get('label')
    meta['label'] = to_json(label)
    meta['columns_'] = to_json(cols)
    for i in range(len(cols)):
        ls = []
        for x, y in value:
            if cols[i] == label:
                ls.append(y.numpy())
            else:
                ls.append(x[i].numpy())
        meta['__values_-key-%d' % i] = to_json(cols[i])
        meta.add_member('__values_-value-%d' % i, builder.run(client, ls))
    meta['__values_-size'] = len(cols)
    meta['partition_index_row_'] = kw.get('partition_index', [0, 0])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [0, 0])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def torch_builder(client, value, builder, **kw):
    typename = kw.get('typename')
    if typename == 'Tensor':
        return torch_tensor_builder(client, value, **kw)
    elif typename == 'Dataframe':
        return torch_dataframe_builder(client, value, builder, **kw)
    else:
        raise TypeError("Only Tensor and Dataframe type supported")


def torch_create_global_tensor(client, value, builder, **kw):
    # TODO
    pass


def torch_create_global_dataframe(client, value, builder, **kw):
    # TODO
    pass


def torch_tensor_resolver(obj):
    meta = obj.meta
    data_shape = from_json(meta['data_shape_'])
    label_shape = from_json(meta['label_shape_'])
    data_name = meta['data_type_']
    label_name = meta['label_type_']
    data_type = normalize_dtype(data_name, meta.get('value_type_meta_', None))
    label_type = normalize_dtype(label_name, meta.get('value_type_meta_', None))
    data = torch.from_numpy(np.frombuffer(memoryview(obj.member('buffer_data_')), dtype=data_type).reshape(data_shape))
    label = torch.from_numpy(
        np.frombuffer(memoryview(obj.member('buffer_label_')), dtype=label_type).reshape(label_shape))
    return torch.utils.data.TensorDataset(data, label)


def torch_dataframe_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        df = resolver(obj, **kw)
    if 'label' in kw:
        target = torch.tensor(df[kw['label']].values.astype(np.float32))
        ds = torch.tensor(df.drop(kw['label'], axis=1).values.astype(np.float32))
        return torch.utils.data.TensorDataset(ds, target)


def torch_record_batch_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        records = resolver(obj, **kw)
    records = records.to_pandas()
    if 'label' in kw:
        target = torch.tensor(records[kw['label']].values)
        ds = torch.tensor(records.drop(kw['label'], axis=1).values)
        return torch.utils.data.TensorDataset(ds, target)


def torch_table_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        table = resolver(obj, **kw)
    table = table.to_pandas()
    if 'label' in kw:
        target = torch.tensor(table[kw['label']].values)
        ds = torch.tensor(table.drop(kw['label'], axis=1).values)
        return torch.utils.data.TensorDataset(ds, target)


def torch_global_tensor_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            data.append(resolver.run(obj.member(f'partitions_{i}')))
    return ConcatDataset(data)


def torch_global_dataframe_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            data.append(resolver.run(obj.member(f'partitions_{i}')))
    return ConcatDataset(data)


def register_torch_types(builder_ctx, resolver_ctx):

    if builder_ctx is not None:
        builder_ctx.register(Dataset, torch_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', torch_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', torch_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', torch_record_batch_resolver)
        resolver_ctx.register('vineyard::Table', torch_table_resolver)
        resolver_ctx.register('vineyard::GlobalTensor', torch_global_tensor_resolver)
        resolver_ctx.register('vineyard::GlobalDataFrame', torch_global_dataframe_resolver)
