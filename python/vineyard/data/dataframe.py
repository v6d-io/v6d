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
import pandas as pd
from pandas.core.internals.blocks import Block
from pandas.core.internals.managers import BlockManager

from vineyard._C import ObjectMeta


def pandas_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    meta['columns_'] = json.dumps(value.columns.values.tolist())
    meta.add_member('index_', builder.run(client, value.index))
    for i, (name, column_value) in enumerate(value.iteritems()):
        np_value = column_value.to_numpy(copy=False)
        meta['__values_-key-%d' % i] = json.dumps(name)
        meta.add_member('__values_-value-%d' % i, builder.run(client, np_value))
    meta['nbytes'] = 0  # FIXME
    meta['__values_-size'] = len(value.columns)
    meta['partition_index_row_'] = kw.get('partition_index', [0, 0])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [0, 0])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def dataframe_resolver(obj, resolver):
    meta = obj.meta
    columns = json.loads(meta['columns_'])
    index = resolver.run(obj.member('index_'))
    if not columns:
        return pd.DataFrame()
    # ensure zero-copy
    blocks = []
    for idx, name in enumerate(columns):
        np_value = resolver.run(obj.member('__values_-value-%d' % idx))
        # ndim: 1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        blocks.append(Block(np.expand_dims(np_value, 0), slice(idx, idx + 1, 1), ndim=2))
    return pd.DataFrame(BlockManager(blocks, [columns, index]))


def register_dataframe_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(pd.DataFrame, pandas_dataframe_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::DataFrame', dataframe_resolver)
