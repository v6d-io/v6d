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

import numpy as np

from vineyard.llm import KVCache
from vineyard.llm import KVTensor
from vineyard.llm.cache import FileCacheConfig
from vineyard.llm.cache import FilesystemType
from vineyard.llm.cache import VineyardCacheConfig


def test_kv_cache_update_and_query_on_blob(vineyard_ipc_sockets):
    vineyard_cache_config = VineyardCacheConfig(
        socket=vineyard_ipc_sockets[0],
        block_size=5,
        sync_interval=3,
        llm_cache_sync_lock="llmCacheSyncLock",
        llm_cache_object_name="llm_cache_object",
        llm_ref_cnt_object_name="llm_refcnt_object",
    )
    cache = KVCache(
        cache_config=vineyard_cache_config,
        tensor_nbytes=16,  # should be the same as the nbytes of the tensor
        cache_capacity=1024,
        layer=2,
    )

    tokens = [1, 2, 3, 4]

    kv_tensors_to_update = []
    kv_tensors = []
    for _ in range(len(tokens)):
        k_tensor = np.random.rand(2, 2).astype(np.float32)
        v_tensor = np.random.rand(2, 2).astype(np.float32)
        kv_tensors.append([(k_tensor, v_tensor) for _ in range(cache.layer)])
        kv_tensors_to_update.append(
            [
                (
                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                )
                for _ in range(cache.layer)
            ]
        )

    # insert the token list and the related kv cache list
    updated = cache.update(None, tokens, kv_tensors_to_update)
    assert updated == len(tokens)

    kv_tensors_to_query = []
    kv_tensors_from_cache = []
    for _ in range(len(tokens)):
        kv_tensors_to_query.append(
            [
                (
                    KVTensor(0, 0),
                    KVTensor(0, 0),
                )
                for _ in range(cache.layer)
            ]
        )

    matched = cache.query(None, tokens, kv_tensors_to_query)
    kv_tensors_from_cache = kv_tensors_to_query[:matched]
    assert matched == len(tokens)

    assert len(kv_tensors) == len(kv_tensors_from_cache)
    for kv, kv_from_cache in zip(kv_tensors, kv_tensors_from_cache):
        assert len(kv) == len(kv_from_cache)
        for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
            kv, kv_from_cache
        ):
            queried_k_tensor = np.frombuffer(
                queried_k_tensor,
                dtype=k_tensor.dtype,
            ).reshape(k_tensor.shape)
            queried_v_tensor = np.frombuffer(
                queried_v_tensor,
                dtype=v_tensor.dtype,
            ).reshape(v_tensor.shape)
            assert np.array_equal(k_tensor, queried_k_tensor)
            assert np.array_equal(v_tensor, queried_v_tensor)


def test_kv_cache_update_and_query_on_fs():
    file_cache_config = FileCacheConfig(
        chunk_size=2,
        hash_chunk_size=2,
        root="/tmp/vineyard/llm_cache",
    )
    cache = KVCache(
        cache_config=file_cache_config,
        tensor_nbytes=16,  # should be the same as the nbytes of the tensor
        cache_capacity=1024,
        layer=2,
    )

    tokens = [1, 2, 3, 4]
    original_kv_tensors = []
    for i in range(0, len(tokens), file_cache_config.chunk_size):
        kv_tensors_to_update = []
        k_tensor = np.random.rand(2, 2).astype(np.float32)
        v_tensor = np.random.rand(2, 2).astype(np.float32)
        for _ in range(file_cache_config.chunk_size):
            original_kv_tensors.append(
                [(k_tensor, v_tensor) for _ in range(cache.layer)]
            )
            kv_tensors_to_update.append(
                [
                    (
                        KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                        KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                    )
                    for _ in range(cache.layer)
                ]
            )
        updated = cache.update(
            tokens[:i],
            tokens[i : i + file_cache_config.chunk_size],
            kv_tensors_to_update,
        )
        assert updated == file_cache_config.chunk_size

    kv_tensors_from_cache = []
    kv_tensors = []
    for _ in range(len(tokens)):
        k_tensor = np.empty((2, 2), dtype=np.float32)
        v_tensor = np.empty((2, 2), dtype=np.float32)
        kv_tensors_from_cache.append([(k_tensor, v_tensor) for _ in range(cache.layer)])
        kv_tensors.append(
            [
                (
                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                )
                for _ in range(cache.layer)
            ]
        )
    matched = cache.query(None, tokens, kv_tensors)
    assert matched == len(tokens)

    assert len(kv_tensors) == len(kv_tensors_from_cache)
    for kv, kv_from_cache in zip(original_kv_tensors, kv_tensors_from_cache):
        assert len(kv) == len(kv_from_cache)
        for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
            kv, kv_from_cache
        ):
            np.array_equal(k_tensor, queried_k_tensor)
            np.array_equal(v_tensor, queried_v_tensor)


def test_kv_cache_update_and_query_on_vineyard_fs(
    vineyard_ipc_sockets, vineyard_endpoints
):
    print(vineyard_endpoints)
    file_cache_config = FileCacheConfig(
        chunk_size=2,
        hash_chunk_size=2,
        root="/tmp/vineyard/llm_cache",
        filesystem_type=FilesystemType.VINEYARD,
        socket=vineyard_ipc_sockets[0],
        rpc_endpoint=vineyard_endpoints[0],
        rdma_endpoint='',
    )
    cache = KVCache(
        cache_config=file_cache_config,
        tensor_nbytes=16,  # should be the same as the nbytes of the tensor
        cache_capacity=1024,
        layer=2,
    )

    tokens = [1, 2, 3, 4]
    original_kv_tensors = []
    kv_tensors_to_update = []
    for _ in range(0, len(tokens), file_cache_config.chunk_size):
        k_tensor = np.random.rand(2, 2).astype(np.float32)
        v_tensor = np.random.rand(2, 2).astype(np.float32)
        for _ in range(file_cache_config.chunk_size):
            original_kv_tensors.append(
                [(k_tensor, v_tensor) for _ in range(cache.layer)]
            )
            kv_tensors_to_update.append(
                [
                    (
                        KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                        KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                    )
                    for _ in range(cache.layer)
                ]
            )

    updated = cache.batched_update(tokens, kv_tensors_to_update)
    assert updated == len(tokens)

    kv_tensors_from_cache = []
    kv_tensors = []
    for _ in range(len(tokens)):
        k_tensor = np.empty((2, 2), dtype=np.float32)
        v_tensor = np.empty((2, 2), dtype=np.float32)
        kv_tensors_from_cache.append([(k_tensor, v_tensor) for _ in range(cache.layer)])
        kv_tensors.append(
            [
                (
                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                )
                for _ in range(cache.layer)
            ]
        )
    matched = cache.batched_query(tokens, kv_tensors)
    assert matched == len(tokens)

    assert len(kv_tensors) == len(kv_tensors_from_cache)
    for kv, kv_from_cache in zip(original_kv_tensors, kv_tensors_from_cache):
        assert len(kv) == len(kv_from_cache)
        for (k_tensor, v_tensor), (queried_k_tensor, queried_v_tensor) in zip(
            kv, kv_from_cache
        ):
            np.array_equal(k_tensor, queried_k_tensor)
            np.array_equal(v_tensor, queried_v_tensor)
