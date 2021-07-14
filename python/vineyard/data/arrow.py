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

import re
import pyarrow as pa

from vineyard._C import ObjectMeta
from .utils import normalize_dtype


def buffer_builder(client, buffer, builder):
    if buffer is None:
        return client.create_empty_blob()
    builder = client.create_blob(len(buffer))
    builder.copy(0, buffer.address, len(buffer))
    return builder.seal(client)


def as_arrow_buffer(blob):
    buffer = blob.buffer
    if buffer is None:
        return None
    return pa.py_buffer(buffer)


def numeric_array_builder(client, array, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::NumericArray<%s>' % array.type
    meta['length_'] = len(array)
    meta['null_count_'] = array.null_count
    meta['offset_'] = array.offset

    null_bitmap = buffer_builder(client, array.buffers()[0], builder)
    buffer = buffer_builder(client, array.buffers()[1], builder)

    meta.add_member('buffer_', buffer)
    meta.add_member('null_bitmap_', null_bitmap)
    meta['nbytes'] = array.nbytes
    return client.create_metadata(meta)


def fixed_size_binary_array_builder(client, array, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::FixedSizeBinaryArray'
    meta['length_'] = len(array)
    meta['null_count_'] = array.null_count
    meta['offset_'] = array.offset
    meta['byte_width_'] = array.byte_width

    null_bitmap = buffer_builder(client, array.buffers()[0], builder)
    buffer = buffer_builder(client, array.buffers()[1], builder)

    meta.add_member('buffer_', buffer)
    meta.add_member('null_bitmap_', null_bitmap)
    meta['nbytes'] = array.nbytes
    return client.create_metadata(meta)


def string_array_builder(client, array, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::LargeStringArray'
    meta['length_'] = len(array)
    meta['null_count_'] = array.null_count
    meta['offset_'] = array.offset

    null_bitmap = buffer_builder(client, array.buffers()[0], builder)
    if isinstance(array, pa.StringArray):
        buffer = array.buffers()[1]
        length = len(buffer) // (pa.uint32().bit_width // 8)
        offset_array = pa.Array.from_buffers(pa.uint32(), length, [None, buffer])
        offset_array = offset_array.cast(pa.uint64())
        offset_buffer = offset_array.buffers()[1]
    else:  # is pa.LargeStringArray
        offset_buffer = array.buffers()[1]
    buffer_offsets = buffer_builder(client, offset_buffer, builder)
    buffer_data = buffer_builder(client, array.buffers()[2], builder)

    meta.add_member('buffer_offsets_', buffer_offsets)
    meta.add_member('buffer_data_', buffer_data)
    meta.add_member('null_bitmap_', null_bitmap)
    meta['nbytes'] = array.nbytes
    return client.create_metadata(meta)


def list_array_builder(client, array, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::LargeListArray'
    meta['length_'] = len(array)
    meta['null_count_'] = array.null_count
    meta['offset_'] = array.offset

    if isinstance(array, pa.ListArray):
        buffer = array.buffers()[1]
        length = len(buffer) // (pa.uint32().bit_width // 8)
        offset_array = pa.Array.from_buffers(pa.uint32(), length, [None, buffer])
        offset_array = offset_array.cast(pa.uint64())
        offset_buffer = offset_array.buffers()[1]
    else:  # is pa.LargeListArray
        offset_buffer = array.buffers()[1]

    meta.add_member('null_bitmap_', buffer_builder(client, array.buffers()[0], builder))
    meta.add_member('buffer_offsets_', buffer_builder(client, offset_buffer, builder))
    meta.add_member('values_', builder.run(client, array.values))
    meta['nbytes'] = array.nbytes
    return client.create_metadata(meta)


def null_array_builder(client, array):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::NullArray'
    meta['length_'] = len(array)
    meta['nbytes'] = 0
    return client.create_metadata(meta)


def boolean_array_builder(client, array, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::BooleanArray'
    meta['length_'] = len(array)
    meta['null_count_'] = array.null_count
    meta['offset_'] = array.offset

    null_bitmap = buffer_builder(client, array.buffers()[0], builder)
    buffer = buffer_builder(client, array.buffers()[1], builder)

    meta.add_member('buffer_', buffer)
    meta.add_member('null_bitmap_', null_bitmap)
    meta['nbytes'] = array.nbytes
    return client.create_metadata(meta)


def schema_proxy_builder(client, schema, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::SchemaProxy'
    serialized = schema.serialize()
    meta.add_member('buffer_', buffer_builder(client, serialized, builder))
    meta['nbytes'] = len(serialized)
    return client.create_metadata(meta)


def record_batch_builder(client, batch, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::RecordBatch'
    meta['row_num_'] = batch.num_rows
    meta['column_num_'] = batch.num_columns
    meta['__columns_-size'] = batch.num_columns

    meta.add_member('schema_', schema_proxy_builder(client, batch.schema, builder))
    for idx in range(batch.num_columns):
        meta.add_member('__columns_-%d' % idx, builder.run(client, batch[idx]))
    meta['nbytes'] = batch.nbytes
    return client.create_metadata(meta)


def table_builder(client, table, builder):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Table'
    meta['num_rows_'] = table.num_rows
    meta['num_columns_'] = table.num_columns
    batches = table.to_batches()
    meta['batch_num_'] = len(batches)
    meta['__batches_-size'] = len(batches)

    meta.add_member('schema_', schema_proxy_builder(client, table.schema, builder))
    for idx, batch in enumerate(batches):
        meta.add_member('__batches_-%d' % idx, record_batch_builder(client, batch, builder))
    meta['nbytes'] = table.nbytes
    return client.create_metadata(meta)


def numeric_array_resolver(obj):
    meta = obj.meta
    typename = obj.typename
    value_type = normalize_dtype(re.match(r'vineyard::NumericArray<([^>]+)>', typename).groups()[0])
    dtype = pa.from_numpy_dtype(value_type)
    buffer = as_arrow_buffer(obj.member('buffer_'))
    null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    length = int(meta['length_'])
    null_count = int(meta['null_count_'])
    offset = int(meta['offset_'])
    return pa.lib.Array.from_buffers(dtype, length, [null_bitmap, buffer], null_count, offset)


def fixed_size_binary_array_resolver(obj):
    meta = obj.meta
    buffer = as_arrow_buffer(obj.member('buffer_'))
    null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    length = int(meta['length_'])
    null_count = int(meta['null_count_'])
    offset = int(meta['offset_'])
    byte_width = int(meta['byte_width_'])
    return pa.lib.Array.from_buffers(pa.binary(byte_width), length, [null_bitmap, buffer], null_count, offset)


def string_array_resolver(obj):
    meta = obj.meta
    buffer_data = as_arrow_buffer(obj.member('buffer_data_'))
    buffer_offsets = as_arrow_buffer(obj.member('buffer_offsets_'))
    null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    length = int(meta['length_'])
    null_count = int(meta['null_count_'])
    offset = int(meta['offset_'])
    return pa.lib.Array.from_buffers(pa.large_string(), length, [null_bitmap, buffer_offsets, buffer_data], null_count,
                                     offset)


def null_array_resolver(obj):
    length = int(obj.meta['length_'])
    return pa.lib.Array.from_buffers(pa.null(), length, [
        None,
    ], length, 0)


def boolean_array_resolver(obj):
    meta = obj.meta
    typename = obj.typename
    buffer = as_arrow_buffer(obj.member('buffer_'))
    null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    length = int(meta['length_'])
    null_count = int(meta['null_count_'])
    offset = int(meta['offset_'])
    return pa.lib.Array.from_buffers(pa.bool_(), length, [null_bitmap, buffer], null_count, offset)


def list_array_resolver(obj, resolver):
    meta = obj.meta
    buffer_offsets = as_arrow_buffer(obj.member('buffer_offsets_'))
    length = int(meta['length_'])
    null_count = int(meta['null_count_'])
    offset = int(meta['offset_'])
    null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    values = resolver.run(obj.member('values_'))
    return pa.lib.Array.from_buffers(pa.large_list(values.type), length, [null_bitmap, buffer_offsets], null_count,
                                     offset, [values])


def schema_proxy_resolver(obj):
    buffer = as_arrow_buffer(obj.member('buffer_'))
    return pa.ipc.read_schema(buffer)


def record_batch_resolver(obj, resolver):
    meta = obj.meta
    nrows, ncolumns = int(meta['row_num_']), int(meta['column_num_'])
    schema = resolver.run(obj.member('schema_'))
    columns = []
    for idx in range(int(meta['__columns_-size'])):
        columns.append(resolver.run(obj.member('__columns_-%d' % idx)))
    return pa.RecordBatch.from_arrays(columns, schema=schema)


def table_resolver(obj, resolver):
    meta = obj.meta
    batch_num = int(meta['batch_num_'])
    batches = []
    for idx in range(int(meta['__batches_-size'])):
        batches.append(resolver.run(obj.member('__batches_-%d' % idx)))
    return pa.Table.from_batches(batches)


def register_arrow_types(builder_ctx=None, resolver_ctx=None):
    if builder_ctx is not None:
        builder_ctx.register(pa.Buffer, buffer_builder)
        builder_ctx.register(pa.NumericArray, numeric_array_builder)
        builder_ctx.register(pa.FixedSizeBinaryArray, fixed_size_binary_array_builder)
        builder_ctx.register(pa.StringArray, string_array_builder)
        builder_ctx.register(pa.LargeStringArray, string_array_builder)
        builder_ctx.register(pa.NullArray, null_array_builder)
        builder_ctx.register(pa.BooleanArray, boolean_array_builder)
        builder_ctx.register(pa.Schema, schema_proxy_builder)
        builder_ctx.register(pa.RecordBatch, record_batch_builder)
        builder_ctx.register(pa.Table, table_builder)
        builder_ctx.register(pa.ListArray, list_array_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::NumericArray', numeric_array_resolver)
        resolver_ctx.register('vineyard::FixedSizeBinaryArray', fixed_size_binary_array_resolver)
        resolver_ctx.register('vineyard::LargeStringArray', string_array_resolver)
        resolver_ctx.register('vineyard::NullArray', null_array_resolver)
        resolver_ctx.register('vineyard::BooleanArray', boolean_array_resolver)
        resolver_ctx.register('vineyard::SchemaProxy', schema_proxy_resolver)
        resolver_ctx.register('vineyard::RecordBatch', record_batch_resolver)
        resolver_ctx.register('vineyard::Table', table_resolver)
        resolver_ctx.register('vineyard::LargeListArray', list_array_resolver)
