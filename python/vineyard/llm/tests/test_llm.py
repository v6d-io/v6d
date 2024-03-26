#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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

import torch

from vineyard.llm import KV_Cache
from vineyard.llm.config import FileCacheConfig
from vineyard.llm.config import VineyardCacheConfig


def test_kv_cache_update_and_query_on_blob(vineyard_ipc_sockets):
    vineyard_cache_config = VineyardCacheConfig(
        socket=vineyard_ipc_sockets[0],
        block_size=5,
        sync_interval=3,
        llm_cache_sync_lock="llmCacheSyncLock",
        llm_cache_object_name="llm_cache_object",
        llm_ref_cnt_object_name="llm_refcnt_object",
    )
    cache = KV_Cache(
        cache_config=vineyard_cache_config,
        tensor_bytes=16,  # should be the same as the nbytes of the tensor
        cache_capacity=10,
        layer=1,
        torch_size=torch.Size([2, 2]),
        dtype=torch.float32,
    )

    kv_cache_list = [
        (torch.rand(2, 2), torch.rand(2, 2)),
        (torch.rand(2, 2), torch.rand(2, 2)),
        (torch.rand(2, 2), torch.rand(2, 2)),
        (torch.rand(2, 2), torch.rand(2, 2)),
    ]

    tokens = [1, 2, 3, 4]
    # insert the token list and the related kv cache list
    cache.update(tokens, kv_cache_list)

    queried_kv_cache_list = cache.query(tokens)

    for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
        kv_cache_list, queried_kv_cache_list
    ):
        assert torch.equal(k_tensor, queried_k_tensor) and torch.equal(
            v_tensor, queried_v_tensor
        )


def test_kv_cache_update_and_query_on_fs():
    file_cache_config = FileCacheConfig(
        batch_size=2,
        split_number=2,
        root="/tmp/vineyard/llm_cache",
    )
    cache = KV_Cache(
        cache_config=file_cache_config,
        tensor_bytes=10000,  # should be the same as the nbytes of the tensor
        cache_capacity=10,
        layer=2,
        torch_size=torch.Size([50, 50]),
        dtype=torch.float32,
    )

    kv_cache_list = [
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
        (torch.rand(50, 50), torch.rand(50, 50)),
    ]

    tokens = [1, 2, 3, 4]
    # insert the token list and the related kv cache list
    cache.update(tokens, kv_cache_list)

    queried_kv_cache_list = cache.query(tokens)

    for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
        kv_cache_list, queried_kv_cache_list
    ):
        assert torch.equal(k_tensor, queried_k_tensor) and torch.equal(
            v_tensor, queried_v_tensor
        )
