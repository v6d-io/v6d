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

import json

import numpy as np
import pandas as pd

import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client

import vineyard
from vineyard.data.dataframe import make_global_dataframe
from vineyard.data.tensor import make_global_tensor


def dask_array_builder(client, value, builder, **kw):
    def put_partition(v, block_id=None):
        client = vineyard.connect()
        obj_id = client.put(v, partition_index=block_id)
        client.persist(obj_id)
        return np.array([[int(obj_id)]])

    _ = Client(kw['dask_scheduler'])  #enforce distributed scheduling
    blocks = value.map_blocks(put_partition, dtype=int).compute().flatten()
    return make_global_tensor(client, blocks)


def dask_dataframe_builder(client, value, builder, **kw):
    def put_partition(v, partition_info=None):
        client = vineyard.connect()
        obj_id = client.put(v, partition_index=(partition_info['number'], 0))
        client.persist(obj_id)
        return pd.DataFrame([{'no': partition_info['number'], 'id': int(obj_id)}])

    _ = Client(kw['dask_scheduler'])  #enforce distributed scheduling
    res = value.map_partitions(put_partition, meta={'no': int, 'id': int}).compute()
    res = res.set_index('no')
    blocks = [res.loc[i] for i in range(len(res))]
    return make_global_dataframe(client, blocks)


def dask_array_resolver(obj, resolver, **kw):
    def get_partition(obj_id):
        client = vineyard.connect()
        np_value = client.get(obj_id)
        return da.from_array(np_value)

    meta = obj.meta
    num = int(meta['partitions_-size'])
    dask_client = Client(kw['dask_scheduler'])
    futures = []
    indices = []
    with_index = True
    for i in range(num):
        ts = meta.get_member('partitions_-%d' % i)
        instance_id = int(ts.meta['instance_id'])

        partition_index = json.loads(ts.meta['partition_index_'])
        if partition_index:
            indices.append((partition_index[0], partition_index[1], i))
        else:
            with_index = False

        futures.append(
            # we require the 1-on-1 alignment of vineyard instances and dask workers.
            # vineyard_sockets maps vineyard instance_ids into ipc_sockets, while
            # dask_workers maps vineyard instance_ids into names of dask workers.
            dask_client.submit(get_partition, ts.meta.id, workers={kw['dask_workers'][instance_id]}))

    arrays = dask_client.gather(futures)
    if with_index:
        indices = list(sorted(indices))
        nx = indices[-1][0] + 1
        ny = indices[-1][1] + 1
        assert nx * ny == num
        rows = []
        for i in range(nx):
            cols = []
            for j in range(ny):
                cols.append(arrays[indices[i * ny + j][2]])
            rows.append(da.hstack(cols))
        return da.vstack(rows)

    return da.vstack(arrays)


def dask_dataframe_resolver(obj, resolver, **kw):
    def get_partition(obj_id):
        client = vineyard.connect()
        df = client.get(obj_id)
        return dd.from_pandas(df, npartitions=1)

    meta = obj.meta
    num = int(meta['partitions_-size'])
    dask_client = Client(kw['dask_scheduler'])
    futures = []
    for i in range(num):
        df = meta.get_member('partitions_-%d' % i)
        instance_id = int(df.meta['instance_id'])
        futures.append(
            # we require the 1-on-1 alignment of vineyard instances and dask workers.
            # vineyard_sockets maps vineyard instance_ids into ipc_sockets, while
            # dask_workers maps vineyard instance_ids into names of dask workers.
            dask_client.submit(get_partition, df.meta.id, workers={kw['dask_workers'][instance_id]}))

    dfs = dask_client.gather(futures)
    return dd.concat(dfs, axis=0)


def register_dask_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(dask.array.Array, dask_array_builder)
        builder_ctx.register(dask.dataframe.DataFrame, dask_dataframe_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::GlobalTensor', dask_array_resolver)
        resolver_ctx.register('vineyard::GlobalDataFrame', dask_dataframe_resolver)
