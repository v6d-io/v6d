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
import pandas as pd
try:
    from pandas.core.internals.blocks import BlockPlacement, NumpyBlock as Block
except ImportError:
    BlockPlacement = None
    from pandas.core.internals.blocks import Block
try:
    from pandas.core.indexes.base import ensure_index
except ImportError:
    try:
        from pandas.core.indexes.base import _ensure_index as ensure_index
    except ImportError:
        from pandas.indexes.base import _ensure_index as ensure_index

try:
    from pandas.core.internals.blocks import DatetimeLikeBlock
except ImportError:
    try:
        from pandas.core.internals.blocks import DatetimeBlock as DatetimeLikeBlock
    except ImportError:
        from pandas.core.internals import DatetimeBlock as DatetimeLikeBlock

try:
    from pandas.core.arrays.datetimes import DatetimeArray
except ImportError:
    DatetimeArray = None

from pandas.core.internals.managers import BlockManager

from vineyard._C import Object, ObjectID, ObjectMeta
from .utils import from_json, to_json, normalize_dtype, expand_slice
from .tensor import ndarray


def pandas_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    meta['columns_'] = to_json(value.columns.values.tolist())
    meta.add_member('index_', builder.run(client, value.index))

    # accumulate columns
    value_columns = [None] * len(value.columns)
    for block in value._mgr.blocks:
        slices = list(expand_slice(block.mgr_locs.indexer))
        if isinstance(block.values, pd.arrays.SparseArray):
            assert len(slices) == 1
            value_columns[slices[0]] = block.values
        elif len(slices) == 1:
            value_columns[slices[0]] = block.values[0]
            vineyard_ref = getattr(block.values, '__vineyard_ref', None)
            # the block comes from vineyard
            if vineyard_ref is not None:
                setattr(value_columns[slices[0]], '__vineyard_ref', vineyard_ref)
        else:
            for index, column_index in enumerate(slices):
                value_columns[column_index] = block.values[index]

    for index, name in enumerate(value.columns):
        meta['__values_-key-%d' % index] = to_json(name)
        meta.add_member('__values_-value-%d' % index, builder.run(client, value_columns[index]))
    meta['nbytes'] = 0  # FIXME
    meta['__values_-size'] = len(value.columns)
    meta['partition_index_row_'] = kw.get('partition_index', [0, 0])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [0, 0])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def pandas_dataframe_resolver(obj, resolver):
    meta = obj.meta
    columns = from_json(meta['columns_'])
    if not columns:
        return pd.DataFrame()
    # ensure zero-copy
    blocks = []
    index_size = 0
    for idx, name in enumerate(columns):
        np_value = resolver.run(obj.member('__values_-value-%d' % idx))
        index_size = len(np_value)
        # ndim: 1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        if BlockPlacement:
            placement = BlockPlacement(slice(idx, idx + 1, 1))
        else:
            placement = slice(idx, idx + 1, 1)
        if DatetimeArray is not None and isinstance(np_value, DatetimeArray):
            values = np_value.reshape(1, -1)
            setattr(values, '__vineyard_ref', getattr(np_value, '__vineyard_ref', None))
            block = DatetimeLikeBlock(values, placement, ndim=2)
        else:
            values = np.expand_dims(np_value, 0).view(ndarray)
            setattr(values, '__vineyard_ref', getattr(np_value, '__vineyard_ref', None))
            block = Block(values, placement, ndim=2)
        blocks.append(block)
    if 'index_' in meta:
        index = resolver.run(obj.member('index_'))
    else:
        index = pd.RangeIndex(index_size)
    return pd.DataFrame(BlockManager(blocks, [ensure_index(columns), index]))


def pandas_sparse_array_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::SparseArray<%s>' % value.dtype.name
    meta['value_type_'] = value.dtype.name
    sp_index_type, (sp_index_size, sp_index_array) = value.sp_index.__reduce__()
    meta['sp_index_name'] = sp_index_type.__name__
    meta['sp_index_size'] = sp_index_size
    meta.add_member('sp_index', builder.run(client, sp_index_array, **kw))
    meta.add_member('sp_values', builder.run(client, value.sp_values, **kw))
    return client.create_metadata(meta)


def pandas_sparse_array_resolver(obj, resolver):
    meta = obj.meta
    value_type = normalize_dtype(meta['value_type_'])
    sp_index_type = getattr(pd._libs.sparse, meta['sp_index_name'])
    sp_index_size = meta['sp_index_size']
    sp_index_array = resolver.run(obj.member('sp_index'))
    sp_index = sp_index_type(sp_index_size, sp_index_array)
    sp_values = resolver.run(obj.member('sp_values'))
    return pd.arrays.SparseArray(sp_values, sparse_index=sp_index, dtype=value_type)


def make_global_dataframe(client, blocks, extra_meta=None):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::GlobalDataFrame'
    meta.set_global(True)
    meta['partitions_-size'] = len(blocks)
    if extra_meta:
        for k, v in extra_meta.items():
            meta[k] = v

    for idx, block in enumerate(blocks):
        if not isinstance(block, (ObjectMeta, ObjectID, Object)):
            block = ObjectID(block)
        meta.add_member('partitions_-%d' % idx, block)

    gtensor_meta = client.create_metadata(meta)
    client.persist(gtensor_meta)
    return gtensor_meta


def register_dataframe_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(pd.DataFrame, pandas_dataframe_builder)
        builder_ctx.register(pd.arrays.SparseArray, pandas_sparse_array_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::DataFrame', pandas_dataframe_resolver)
        resolver_ctx.register('vineyard::SparseArray', pandas_sparse_array_resolver)
