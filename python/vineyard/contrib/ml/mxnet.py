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

import mxnet as mx

import numpy as np

from vineyard._C import ObjectMeta
from vineyard.core.resolver import resolver_context, default_resolver_context
from vineyard.data.utils import from_json, to_json, build_numpy_buffer, normalize_dtype


def mxnet_tensor_builder(client, value, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Tensor'
    meta['partition_index_'] = to_json(kw.get('partition_index', []))
    data = mx.gluon.data.DataLoader(value, batch_size=len(value))
    for x, y in data:
        meta.add_member('buffer_data_', build_numpy_buffer(client, x.asnumpy()))
        meta.add_member('buffer_label_', build_numpy_buffer(client, y.asnumpy()))
        meta['data_shape_'] = to_json(x.asnumpy().shape)
        meta['label_shape_'] = to_json(y.asnumpy().shape)
        meta['data_type_'] = x.asnumpy().dtype.name
        meta['label_type_'] = y.asnumpy().dtype.name
        meta['data_type_meta_'] = x.asnumpy().dtype.str
        meta['label_type_meta_'] = y.asnumpy().dtype.str
    return client.create_metadata(meta)


def mxnet_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    cols = kw.get('cols')
    label = kw.get('label')
    meta['label'] = to_json(label)
    meta['columns_'] = to_json(cols)
    meta['__values_-key-%d' % (len(cols) - 1)] = to_json(label)
    meta.add_member('__values_-value-%d' % (len(cols) - 1), builder.run(client, value[1]))
    for i in range(len(cols) - 1):
        meta['__values_-key-%d' % i] = to_json(cols[i])
        meta.add_member('__values_-value-%d' % i, builder.run(client, value[0][:, i]))
    meta['__values_-size'] = len(cols)
    meta['partition_index_row_'] = kw.get('partition_index', [0, 0])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [0, 0])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def mxnet_builder(client, value, builder, **kw):
    typename = kw.get('typename')
    if typename == 'Tensor':
        return mxnet_tensor_builder(client, value, **kw)
    elif typename == 'DataFrame':
        return mxnet_dataframe_builder(client, value, builder, **kw)


def mxnet_tensor_resolver(obj, resolver, **kw):
    meta = obj.meta
    data_shape = from_json(meta['data_shape_'])
    label_shape = from_json(meta['label_shape_'])
    data_name = meta['data_type_']
    label_name = meta['label_type_']
    data_type = normalize_dtype(data_name, meta.get('value_type_meta_', None))
    label_type = normalize_dtype(label_name, meta.get('value_type_meta_', None))
    data = np.frombuffer(memoryview(obj.member('buffer_data_')), dtype=data_type).reshape(data_shape)
    label = np.frombuffer(memoryview(obj.member('buffer_label_')), dtype=label_type).reshape(label_shape)
    return mx.gluon.data.ArrayDataset((data, label))


def mxnet_dataframe_resolver(obj, resolver, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        df = resolver(obj, **kw)
    if 'label' in kw:
        target = df[kw['label']].values.astype(np.float32)
        data = df.drop(kw['label'], axis=1).values.astype(np.float32)
        return mx.gluon.data.ArrayDataset((data, target))


def mxnet_record_batch_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        records = resolver(obj, **kw)
    records = records.to_pandas()
    if 'label' in kw:
        target = records[kw['label']].values.astype(np.float32)
        data = records.drop(kw['label'], axis=1).values.astype(np.float32)
        return mx.gluon.data.ArrayDataset((data, target))


def mxnet_table_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        table = resolver(obj, **kw)
    table = table.to_pandas()
    if 'label' in kw:
        target = table[kw['label']].values.astype(np.float32)
        data = table.drop(kw['label'], axis=1).values.astype(np.float32)
        return mx.gluon.data.ArrayDataset((data, target))


def mxnet_global_tensor_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    label = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            temp = resolver.run(obj.member(f'partitions_{i}'))
            data.append(temp[0])
            label.append(temp[1])
    return mx.gluon.data.ArrayDataset((data, label))


def mxnet_global_dataframe_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    label = []
    for i in range(num):
        if meta[f'partitions_{i}'].islocal:
            temp = resolver.run(obj.member(f'partitions_{i}'))
            data.append(temp[0])
            label.append(temp[1])
    return mx.gluon.data.ArrayDataset((data, label))


def register_mxnet_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(mx.gluon.data.ArrayDataset, mxnet_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', mxnet_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', mxnet_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', mxnet_record_batch_resolver)
        resolver_ctx.register('vineyard::Table', mxnet_table_resolver)
        resolver_ctx.register('vineyard::GlobalTensor', mxnet_global_tensor_resolver)
        resolver_ctx.register('vineyard::GlobalDataFrame', mxnet_global_dataframe_resolver)
