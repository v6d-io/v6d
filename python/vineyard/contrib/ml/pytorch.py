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

from vineyard._C import ObjectMeta
from vineyard.data.utils import from_json, to_json, build_numpy_buffer, normalize_dtype

import pandas as pd
import pyarrow as pa
try:
    from pandas.core.internals.blocks import BlockPlacement, NumpyBlock as Block
except:
    BlockPlacement = None
    from pandas.core.internals.blocks import Block

from pandas.core.internals.managers import BlockManager
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
    meta['columns_'] = to_json(cols)
    for i in range(len(cols)):
        ls = []
        for x, y in value:
            if cols[i] == 'target':
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
    dataset = torch.utils.data.TensorDataset(data, label)
    return dataset


def torch_dataframe_resolver(obj, resolver):
    meta = obj.meta
    columns = from_json(meta['columns_'])
    if not columns:
        return pd.DataFrame()
    blocks = []
    index_size = 0
    for idx, name in enumerate(columns):
        np_value = resolver.run(obj.member('__values_-value-%d' % idx))
        index_size = len(np_value)
        if BlockPlacement:
            placement = BlockPlacement(slice(idx, idx + 1, 1))
        else:
            placement = slice(idx, idx + 1, 1)
        values = np.expand_dims(np_value, 0)
        blocks.append(Block(values, placement, ndim=2))
    if 'index_' in meta:
        index = resolver.run(obj.member('index_'))
    else:
        index = np.arange(index_size)
    df = pd.DataFrame(BlockManager(blocks, [pd.Index(columns), index]))
    target = torch.tensor(df['target'].values.astype(np.float32))
    ds = torch.tensor(df.drop('target', axis=1).values.astype(np.float32))
    return torch.utils.data.TensorDataset(ds, target)


def torch_recordBatch_resolver(obj, resolver):
    meta = obj.meta
    schema = resolver.run(obj.member('schema_'))
    columns = []
    for idx in range(int(meta['__columns_-size'])):
        columns.append(resolver.run(obj.member('__columns_-%d' % idx)))
    arrow = pa.RecordBatch.from_arrays(columns, schema=schema).to_pandas()
    target = torch.tensor(arrow['target'].values)
    ds = torch.tensor(arrow.drop('target', axis=1).values)
    return torch.utils.data.TensorDataset(ds, target)


def torch_table_resolver(obj, resolver):
    meta = obj.meta
    batches = []
    for idx in range(int(meta['__batches_-size'])):
        batches.append(resolver.run(obj.member('__batches_-%d' % idx)))
    arrow = pa.Table.from_batches(batches).to_pandas()
    target = torch.tensor(arrow['target'].values)
    ds = torch.tensor(arrow.drop('target', axis=1).values)
    return torch.utils.data.TensorDataset(ds, target)


def register_tensor_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(Dataset, torch_tensor_builder)
        builder_ctx.register(Dataset, torch_dataframe_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', torch_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', torch_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', torch_recordBatch_resolver)
        resolver_ctx.register('vineyard::Table', torch_table_resolver)
