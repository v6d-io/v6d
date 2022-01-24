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
from ray.data.block import Block
from ray.data.block import BlockAccessor
from ray.data.dataset import Dataset
from ray.data.impl.block_list import BlockList
from ray.data.impl.remote_fn import cached_remote_fn

import vineyard
from vineyard.data.dataframe import make_global_dataframe

from .actor import spread
from .actor import spread_and_get
from .actor import spread_to_all_nodes


def _block_to_vineyard(block: Block):
    client = vineyard.connect()
    block = BlockAccessor.for_block(block)
    return client.put(block.to_pandas())


def to_vineyard(self):
    client = vineyard.connect()
    block_to_vineyard = cached_remote_fn(_block_to_vineyard, num_cpus=0.1)
    blocks = ray.get([block_to_vineyard.remote(block) for block in self._blocks])
    return make_global_dataframe(client, blocks).id


def _vineyard_to_block(object_id):
    client = vineyard.connect()
    df = client.get(object_id)
    block = pa.table(df)
    return (block, BlockAccessor.for_block(block).get_metadata(input_files=None))


def _get_remote_chunks_map(object_id):
    client = vineyard.connect()
    meta = client.get_meta(object_id)
    if meta.typename == "vineyard::DataFrame" or meta.typename.startswith(
        "vineyard::Tensor"
    ):
        return {repr(object_id): meta.instance_id}

    if (
        meta.typename == "vineyard::GlobalDataFrame"
        or meta.typename == "vineyard::GlobalTensor"
    ):
        mapping = dict()
        for index in range(int(meta['partitions_-size'])):
            item = meta['partitions_-%d' % index]
            mapping[repr(item.id)] = item.instance_id
        return mapping

    raise NotImplementedError("Not implemented: blocks from %s" % meta.typename)


def _get_vineyard_instance_id():
    client = vineyard.connect()
    return client.instance_id


def from_vineyard(object_id):
    vineyard_to_block = cached_remote_fn(
        _vineyard_to_block, num_cpus=0.1, num_returns=2
    )
    get_vineyard_instance_id = cached_remote_fn(_get_vineyard_instance_id, num_cpus=0.1)
    get_remote_chunks_map = cached_remote_fn(_get_remote_chunks_map, num_cpus=0.1)

    chunks = ray.get(get_remote_chunks_map.remote(object_id))

    with spread_to_all_nodes(get_vineyard_instance_id) as (nodes, pg):
        instances = dict()  # instance_id -> placement group index
        for index in range(nodes):
            instance = ray.get(
                get_vineyard_instance_id.options(
                    placement_group=pg, placement_group_bundle_index=index
                ).remote()
            )
            instances[instance] = index

        blocks, metadatas = [], []
        for object_id, location in chunks.items():
            block, metadata = vineyard_to_block.options(
                placement_group=pg, placement_group_bundle_index=instances[location]
            ).remote(vineyard.ObjectID(object_id))
            blocks.append(block)
            metadatas.append(metadata)

        return Dataset(BlockList(blocks, ray.get(metadatas)))


def __inject_to_dataset():
    setattr(Dataset, 'to_vineyard', to_vineyard)
    setattr(ray.data, 'from_vineyard', from_vineyard)
    setattr(ray.data.read_api, 'from_vineyard', from_vineyard)


__inject_to_dataset()
del __inject_to_dataset
