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

import contextlib
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ._llm_C import FilesystemType
from ._llm_C import KVCacheManager
from ._llm_C import KVTensor

logger = logging.getLogger('vineyard')


def _argument_from_env(
    kwargs: Dict[str, Any],
    envprefix: str,
    name: str,
    dtype=None,
):
    envname = f'{envprefix}_{name.upper()}'
    if envname in os.environ:
        value = os.environ.get(envname)
        if dtype:
            value = dtype(value)
        kwargs[name] = value


class VineyardCacheConfig:
    """VineyardCacheConfig is a class to configure the llm kv cache in vineyard."""

    def __init__(
        self,
        socket: str,
        block_size: int = 5,
        sync_interval: int = 3,
        llm_cache_sync_lock: str = "llmCacheSyncLock",
        llm_cache_object_name: str = "llm_cache_object",
        llm_ref_cnt_object_name: str = "llm_refcnt_object",
    ):
        """Create a vineyard cache config.

        Args:
            socket (str):
                The ipc socket of the vineyardd instance.
            block_size (int, optional):
                The block size of the kv cache. Defaults to 5.
            sync_interval (int, optional):
                The sync interval of the kv cache. Defaults to 3.
            llm_cache_sync_lock (str, optional):
                The name of the kv cache sync lock. Defaults to "llmCacheSyncLock".
            llm_cache_object_name (str, optional):
                The name of the kv cache object. Defaults to "llm_cache_object".
            llm_ref_cnt_object_name (str, optional):
                The name of the kv cache ref cnt object.
                Defaults to "llm_refcnt_object".
        """
        import vineyard

        self.block_size = block_size
        self.sync_interval = sync_interval
        self.llm_cache_sync_lock = llm_cache_sync_lock
        self.llm_cache_object_name = llm_cache_object_name
        self.llm_ref_cnt_object_name = llm_ref_cnt_object_name

        # Connecting to vineyardd
        self.ipc_client = vineyard.connect(socket).ipc_client

    def __repr__(self):
        return (
            f'VineyardCacheConfig('
            f'ipc_client={self.ipc_client}, '
            f'block_size={self.block_size}, '
            f'sync_interval={self.sync_interval}, '
            f'llm_cache_sync_lock={self.llm_cache_sync_lock}, '
            f'llm_cache_object_name={self.llm_cache_object_name}, '
            f'llm_ref_cnt_object_name={self.llm_ref_cnt_object_name})'
        )


class FileCacheConfig:
    """FileCacheConfig is a class to configure the llm kv cache on filesystem."""

    def __init__(
        self,
        chunk_size: int = 16,
        hash_chunk_size: int = 2,
        root: str = "/tmp/vineyard/llm_cache",
        filesystem_type: FilesystemType = FilesystemType.LOCAL,
        gc_interval: int = 30 * 60,
        ttl: int = 30 * 60,
        enable_global_gc: bool = False,
        global_gc_interval: int = 3 * 60 * 60,
        global_ttl: int = 3 * 60 * 60,
        socket: str = "",
        rpc_endpoint: str = "",
        rdma_endpoint: str = "",
    ):
        """Create a file cache config.

        Args:
            chunk_size (int):
                Divide the token list into batches, each batch
                contains chunk_size tokens. Defaults to 16.
            hash_chunk_size (int):
                Split the hash value into the file with multiple directories.
                e.g, hash_chunk_size=2, hash value=123456, the file path is 12/34/56.
            root (str):
                The root directory of the kv state files.
                Defaults to "/tmp/vineyard/llm_cache".
            filesystem_type (str):
                The type of the filesystem. Defaults to "local".
            gc_interval (int):
                The interval of the client gc (seconds).
                Defaults to 30 * 60 seconds.
            ttl (int):
                The time to live of the kv state files (seconds).
                Defaults to 30 * 60 seconds.
            enable_global_gc (bool):
                Enable the global gc or not. Defaults to False.
            global_gc_interval (int):
                The interval of the global gc (seconds).
                Defaults to 3 * 60 * 60 seconds.
            global_ttl (int):
                The time to live of the global gc files (seconds).
                Defaults to 3 * 60 * 60 seconds.
        """
        self.chunk_size = chunk_size
        self.hash_chunk_size = hash_chunk_size
        self.root = root
        self.filesystem_type = filesystem_type
        self.gc_interval = gc_interval
        self.ttl = ttl
        self.enable_global_gc = enable_global_gc
        self.global_gc_interval = global_gc_interval
        self.global_ttl = global_ttl

        import vineyard

        if filesystem_type == FilesystemType.VINEYARD:
            self.ipc_client = vineyard.connect(socket).ipc_client
            rpc_host = rpc_endpoint.split(":")[0]
            rpc_port = rpc_endpoint.split(":")[1]
            self.rpc_client = vineyard.connect(
                host=rpc_host, port=rpc_port, rdma_endpoint=rdma_endpoint
            ).rpc_client

    def __repr__(self):
        return (
            f'FileCacheConfig('
            f'chunk_size={self.chunk_size}, '
            f'hash_chunk_size={self.hash_chunk_size}, '
            f'root={self.root}, '
            f'filesystem_type={self.filesystem_type}, '
            f'gc_interval={self.gc_interval}, '
            f'ttl={self.ttl}, '
            f'enable_global_gc={self.enable_global_gc}, '
            f'global_gc_interval={self.global_gc_interval}, '
            f'global_ttl={self.global_ttl}), '
        )


class KVCache:  # pylint: disable=too-many-instance-attributes
    """KVCache is a class that manages the llm kv cache in vineyard."""

    def __init__(
        self,
        cache_config: Optional[Union[VineyardCacheConfig, FileCacheConfig]] = None,
        tensor_nbytes: int = 1024,
        cache_capacity: int = 1024,
        layer: int = 1,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        **kwargs,
    ):
        """Create a llm kv cache manager based on vineyard blob.

        Args:
            cache_config (Union[VineyardCacheConfig, FileCacheConfig]):
                The config of the KV cache, including vineyard cache and file cache.
            tensor_nbytes (int, optional):
                The size of the k/v cache tensor for each token at each layer.
                Defaults to 10.
            cache_capacity (int, optional):
                The capacity of the KV cache refers to the maximum number of
                tokens it can hold. Defaults to 10.
            layer (int, optional):
                The number of layers of the kv cache. Defaults to 1.
            rank (int, optional):
                The rank of the current worker. Defaults to None.
        """
        self.kv_cache_manager = None

        if cache_config is None:
            if 'VINEYARD_LLM_CACHE_SHARED_MEMORY' in os.environ:
                config = {}
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_SHARED_MEMORY', 'socket', str
                )
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_SHARED_MEMORY', 'block_size', int
                )
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_SHARED_MEMORY', 'sync_interval', int
                )
                cache_config = VineyardCacheConfig(**config)
            if 'VINEYARD_LLM_CACHE_FILESYSTEM' in os.environ:
                config = {}
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_FILESYSTEM', 'chunk_size', int
                )
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_FILESYSTEM', 'hash_chunk_size', int
                )
                _argument_from_env(
                    config, 'VINEYARD_LLM_CACHE_FILESYSTEM', 'root', dtype=str
                )
                cache_config = FileCacheConfig(**config)

        if rank is not None and world_size is not None:
            if isinstance(cache_config, FileCacheConfig):
                cache_config.root = os.path.join(
                    cache_config.root, f'{world_size}-{rank}'
                )

        logger.info("Initializing vineyard llm cache with config: %r", cache_config)
        if not isinstance(cache_config, VineyardCacheConfig) and not isinstance(
            cache_config, FileCacheConfig
        ):
            raise ValueError(
                "The cache_config should be VineyardCacheConfig or FileCacheConfig."
            )
        self.cache_config = cache_config
        self.tensor_nbytes = tensor_nbytes
        self.cache_capacity = cache_capacity
        self.layer = layer

        self.kv_cache_manager = KVCacheManager(
            tensor_nbytes=tensor_nbytes,
            cache_capacity=cache_capacity,
            layer=layer,
            **cache_config.__dict__,
            **kwargs,
        )
        if isinstance(cache_config, VineyardCacheConfig):
            self.chunk_size = cache_config.block_size
        else:
            self.chunk_size = cache_config.chunk_size

    def __repr__(self):
        return (
            'KVCache('
            f'cache_config={self.cache_config}, '
            f'tensor_nbytes={self.tensor_nbytes}, '
            f'cache_capacity={self.cache_capacity}, '
            f'layer={self.layer})'
        )

    def update(
        self,
        prefix: List[int],
        tokens: List[int],
        kv_cache_list: List[List[Tuple[KVTensor, KVTensor]]],
    ) -> int:
        """Update the kv cache stored in vineyard.

        Args:
            prefix (list): the prefix of the tokens
                For FileCacheConfig, the length of the prefix should be
                multiple of the chunk size.
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]
            kv_cache_list (List[List[Tuple[KVTensor, KVTensor]]]):
                the kv tensors list of the related tokens including all layers, and
                its length should be the same as the length of tokens.

                The k, v tensor for i-th token at the j-th layer is: kv_cache_list[i][j]

                Whether the underlying kv cache is vineyard or file, the
                kv_cache_list is managed by the caller.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_cache_list as follows:

                .. code:: python

                    kv_cache_list = []
                    for _ in range(2): # the number of tokens
                        k_tensor = np.random.rand(2,2).astype(np.float32)
                        v_tensor = np.random.rand(2,2).astype(np.float32)
                        kv_cache_list.append(
                            [
                                (
                                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                                )
                                for _ in range(2) # the number of layers
                            ]
                        )

        """
        if prefix:
            return self.kv_cache_manager.update(prefix, tokens, kv_cache_list)
        else:
            return self.kv_cache_manager.update(tokens, kv_cache_list)

    def batched_update(
        self,
        tokens: List[int],
        kv_cache_list: List[List[Tuple[KVTensor, KVTensor]]],
    ) -> int:
        return self.kv_cache_manager.batched_update(tokens, kv_cache_list)

    def query(
        self,
        prefix: List[int],
        tokens: List[int],
        kv_cache_list: List[List[Tuple[KVTensor, KVTensor]]],
    ) -> int:
        """Query the kv cache stored in vineyard.

        Args:
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]
            kv_cache_list: (List[List[Tuple[KVTensor, KVTensor]]]):
                the kv tensors list of the related tokens including all layers, and its
                length should be the same as the length of tokens.

                The k, v tensor for i-th token at the j-th layer is: kv_cache_list[i][j]

                For VineyardConfigCache, the kv_cache_list is managed by vineyard.
                The caller does not need to malloc and free the memory of the kv state.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_cache_list as follows:

                .. code:: python

                    kv_cache_list = [
                        (
                            KVTensor(0, 0),
                            KVTensor(0, 0),
                        ) for _ in range(2) # the number of layers
                    ] * 2 # the number of tokens

                For FileCacheConfig, the kv_cache_list is managed by the caller.
                The caller needs to malloc and free the memory of the kv state.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_cache_list as follows:

                .. code:: python

                    kv_cache_list = []
                    for _ in range(2): # the number of tokens
                        k_tensor = np.empty((2,2), dtype=np.float32)
                        v_tensor = np.empty((2,2), dtype=np.float32)
                        kv_cache_list.append(
                            [
                                (
                                    KVTensor(k_tensor.ctypes.data, k_tensor.nbytes),
                                    KVTensor(v_tensor.ctypes.data, v_tensor.nbytes),
                                )
                                for _ in range(2) # the number of layers
                            ]
                        )

        Returns:
            int: The number of matched tokens.
        """
        if prefix:
            return self.kv_cache_manager.query(prefix, tokens, kv_cache_list)
        else:
            return self.kv_cache_manager.query(tokens, kv_cache_list)

    def batched_query(
        self,
        tokens: List[int],
        kv_cache_list: List[List[Tuple[KVTensor, KVTensor]]],
    ) -> int:
        return self.kv_cache_manager.batched_query(tokens, kv_cache_list)

    def __del__(self):
        if self.kv_cache_manager:
            with contextlib.suppress(Exception):
                self.kv_cache_manager.close()
