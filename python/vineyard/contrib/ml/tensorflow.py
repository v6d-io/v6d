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
import tensorflow as tf

from vineyard._C import ObjectMeta
from vineyard.core.resolver import resolver_context, default_resolver_context
from vineyard.data.utils import from_json, to_json, build_numpy_buffer, normalize_dtype


def tf_tensor_builder(client, value, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Tensor'
    meta['num'] = to_json(len(value))
    meta['partition_index_'] = to_json(kw.get('partition_index', []))
    data = value
    data = value.batch(len(value))
    for i in data:
        meta.add_member('buffer_data_', build_numpy_buffer(client, i[0].numpy()))
        meta.add_member('buffer_label_', build_numpy_buffer(client, i[1].numpy()))
        meta['data_shape_'] = to_json(i[0].numpy().shape)
        meta['label_shape_'] = to_json(i[1].numpy().shape)
        meta['data_type_'] = i[0].numpy().dtype.name
        meta['label_type_'] = i[1].numpy().dtype.name
        meta['data_type_meta_'] = i[0].numpy().dtype.str
        meta['label_type_meta_'] = i[1].numpy().dtype.str
    return client.create_metadata(meta)


def tf_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    for feat, labels in value.take(1):
        cols = list(feat.keys())
    cols.append('label')
    meta['columns_'] = to_json(cols)
    for i in range(len(cols)):
        ls = []
        for feat, labels in value.take(len(value)):
            if cols[i] == 'label':
                ls.append(labels.numpy())
            else:
                ls.append(feat[cols[i]].numpy())
        meta['__values_-key-%d' % i] = to_json(cols[i])
        meta.add_member('__values_-value-%d' % i, builder.run(client, ls))
    meta['__values_-size'] = len(cols)
    meta['partition_index_row_'] = kw.get('partition_index', [0, 0])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [0, 0])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def tf_builder(client, value, builder, **kw):
    for x, y in value.take(1):
        if isinstance(x, dict):
            return tf_dataframe_builder(client, value, builder, **kw)
        else:
            return tf_tensor_builder(client, value, **kw)


def tf_create_global_tensor(client, value, builder, **kw):
    # TODO
    pass


def tf_create_global_dataframe(client, value, builder, **kw):
    # TODO
    pass


def tf_tensor_resolver(obj):
    meta = obj.meta
    num = from_json(meta['num'])
    data_shape = from_json(meta['data_shape_'])
    label_shape = from_json(meta['label_shape_'])
    data_name = meta['data_type_']
    label_name = meta['label_type_']
    data_type = normalize_dtype(data_name, meta.get('value_type_meta_', None))
    label_type = normalize_dtype(label_name, meta.get('value_type_meta_', None))
    data = np.frombuffer(memoryview(obj.member('buffer_data_')), dtype=data_type).reshape(data_shape)
    label = np.frombuffer(memoryview(obj.member('buffer_label_')), dtype=label_type).reshape(label_shape)
    data = tf.data.Dataset.from_tensor_slices((data, label))
    return data


def tf_dataframe_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        df = resolver(obj, **kw)
    labels = df.pop(kw.get('label', 'label'))
    if 'data' in kw:
        return tf.data.Dataset.from_tensor_slices((np.stack(df[kw['data']], axis=0), labels))
    return tf.data.Dataset.from_tensor_slices((dict(df), labels))


def tf_record_batch_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        records = resolver(obj, **kw)
    records = records.to_pandas()
    labels = records.pop('label')
    return tf.data.Dataset.from_tensor_slices((dict(records), labels))


def tf_table_resolver(obj, resolver):
    meta = obj.meta
    batches = []
    for idx in range(int(meta['__batches_-size'])):
        batches.append(resolver(obj.member('__batches_-%d' % idx)))
    tf_data = batches[0]
    for i in range(1, len(batches)):
        tf_data = tf_data.concatenate(batches[i])
    return tf_data


def tf_global_tensor_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_-{i}'].islocal:
            data.append(resolver.run(obj.member(f'partitions_-{i}')))
    tf_data = data[0]
    for i in range(1, len(data)):
        tf_data = tf_data.concatenate(data[i])
    return tf_data


def tf_global_dataframe_resolver(obj, resolver, **kw):
    meta = obj.meta
    num = int(meta['partitions_-size'])
    data = []
    for i in range(num):
        if meta[f'partitions_-{i}'].islocal:
            data.append(resolver(obj.member(f'partitions_-{i}'), **kw))
    tf_data = data[0]
    for i in range(1, len(data)):
        tf_data = tf_data.concatenate(data[i])
    return tf_data


def register_tf_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(tf.data.Dataset, tf_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', tf_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', tf_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', tf_record_batch_resolver)
        resolver_ctx.register('vineyard::Table', tf_table_resolver)
        resolver_ctx.register('vineyard::GlobalTensor', tf_global_tensor_resolver)
        resolver_ctx.register('vineyard::GlobalDataFrame', tf_global_dataframe_resolver)
