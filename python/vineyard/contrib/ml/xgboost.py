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
from vineyard.data import tensor, dataframe, arrow

import pandas as pd
import pyarrow as pa
try:
    from pandas.core.internals.blocks import BlockPlacement, NumpyBlock as Block
except:
    BlockPlacement = None
    from pandas.core.internals.blocks import Block

from pandas.core.internals.managers import BlockManager
import numpy as np
import xgboost as xgb


def xgb_builder(client, value, builder, **kw):
    # TODO: build DMatrix to vineyard objects
    pass


def xgb_tensor_resolver(obj):
    array = tensor.numpy_ndarray_resolver(obj)
    return xgb.DMatrix(array)


def xgb_dataframe_resolver(obj, resolver, **kw):
    meta = obj.meta
    columns = from_json(meta['columns_'])
    if not columns:
        return pd.DataFrame()
    # ensure zero-copy
    blocks = []
    index_size = 0
    for idx, name in enumerate(columns):
        np_value = tensor.numpy_ndarray_resolver(obj.member('__values_-value-%d' % idx))
        index_size = len(np_value)
        # ndim: 1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        if BlockPlacement:
            placement = BlockPlacement(slice(idx, idx + 1, 1))
        else:
            placement = slice(idx, idx + 1, 1)
        values = np.expand_dims(np_value, 0)
        blocks.append(Block(values, placement, ndim=2))
    index = np.arange(index_size)
    df = pd.DataFrame(BlockManager(blocks, [pd.Index(columns), index]))
    if 'label' in kw:
        label = df.pop(kw['label'])
        # data column can only be specified if label column is specified
        if 'data' in kw:
            df = np.stack(df[kw['data']].values)
        return xgb.DMatrix(df, label)
    return xgb.DMatrix(df)


def xgb_recordBatch_resolver(obj, resolver, **kw):
    rb = arrow.record_batch_resolver(obj, resolver)
    # FIXME to_pandas is not zero_copy guaranteed
    df = rb.to_pandas()
    if 'label' in kw:
        label = df.pop(kw['label'])
        return xgb.DMatrix(df, label)
    return xgb.DMatrix(df)


def xgb_table_resolver(obj, resolver, **kw):
    meta = obj.meta
    batch_num = int(meta['batch_num_'])
    batches = []
    for idx in range(int(meta['__batches_-size'])):
        batches.append(arrow.record_batch_resolver(obj.member('__batches_-%d' % idx), resolver))
    tb = pa.Table.from_batches(batches)
    # FIXME to_pandas is not zero_copy guaranteed
    df = tb.to_pandas()
    if 'label' in kw:
        label = df.pop(kw['label'])
        return xgb.DMatrix(df, label)
    return xgb.DMatrix(df)


def register_xgb_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(xgb.DMatrix, xgb_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', xgb_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', xgb_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', xgb_recordBatch_resolver)
        resolver_ctx.register('vineyard::Table', xgb_table_resolver)
