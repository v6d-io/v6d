#!/usr/bin/env python3
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

import pyarrow as pa

import ray
import ray.data
import ray.data.read_api
from ray.data.dataset import Dataset
from ray.data.block import Block, BlockAccessor
from ray.data.impl.block_list import BlockList
from ray.data.impl.remote_fn import cached_remote_fn

import vineyard
from vineyard.data.dataframe import make_global_dataframe

from .actor import spread


def _block_to_vineyard(block: Block):
    client = vineyard.connect()
    block = BlockAccessor.for_block(block)
    return client.put(block.to_pandas())


def _vineyard_tensor_to_block(client, object_id):
    raise NotImplementedError


def _vineyard_dataframe_to_block(client, object_id):
    df = client.get(object_id)
    block = pa.table(df)
    return (block, BlockAccessor.for_block(block).get_metadata(input_files=None))


def _vineyard_to_blocks(object_id):
    client = vineyard.connect()
    meta = client.get_meta(object_id)
    if meta.typename == 'vineyard::DataFrame':
        return [_vineyard_dataframe_to_block(client, object_id)]

    if meta.typename == 'vineyard::GlobalDataFrame':
        blocks = []
        for index in range(int(meta['partitions_-size'])):
            df = meta.get_number('partitions_-%d' % index)
            if df.instance_id == client.instance_id:
                blocks.append(_vineyard_dataframe_to_block(client, df.id))
        return blocks

    if meta.typename.starts('vineyard::Tensor'):
        return [_vineyard_tensor_to_block(client, object_id)]

    if meta.typename == 'vineyard::GlobalTensor':
        blocks = []
        for index in range(int(meta['partitions_-size'])):
            df = meta.get_number('partitions_-%d' % index)
            if df.instance_id == client.instance_id:
                blocks.append(_vineyard_tensor_to_block(client, df.id))
        return blocks

    raise NotImplementedError("Not implemented: blocks from %s" % meta.typename)


def to_vineyard(self):
    client = vineyard.connect()
    block_to_vineyard = cached_remote_fn(_block_to_vineyard, num_cpus=0.1)
    blocks = ray.get([block_to_vineyard.remote(block) for block in self._blocks])
    return make_global_dataframe(client, blocks).id


def from_vineyard(object_id):
    vineyard_to_blocks = cached_remote_fn(_vineyard_to_blocks, num_cpus=0.1)
    results = spread(vineyard_to_blocks, object_id)
    blocks, metadata = [], []
    for res in results:
        blocks.extend(res[0])
        metadata.extend(res[1])
    return Dataset(BlockList(blocks, ray.get(metadata)))


def __inject_to_dataset():
    setattr(Dataset, 'to_vineyard', to_vineyard)
    setattr(ray.data, 'from_vineyard', from_vineyard)
    setattr(ray.data.read_api, 'from_vineyard', from_vineyard)


__inject_to_dataset()
del __inject_to_dataset
