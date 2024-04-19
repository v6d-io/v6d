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

import vineyard

from .llm_C import FilesystemType


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
        self.ipc_client = vineyard.connect(socket).ipc_client
        self.block_size = block_size
        self.sync_interval = sync_interval
        self.llm_cache_sync_lock = llm_cache_sync_lock
        self.llm_cache_object_name = llm_cache_object_name
        self.llm_ref_cnt_object_name = llm_ref_cnt_object_name


class FileCacheConfig:
    """FileCacheConfig is a class to configure the llm kv cache on filesystem."""

    def __init__(
        self,
        chunk_size: int = 16,
        split_number: int = 2,
        root: str = "/tmp/vineyard/llm_cache",
        filesystem_type: FilesystemType = FilesystemType.LOCAL,
        client_gc_interval: int = 30 * 60,
        ttl: int = 30 * 60,
        enable_global_gc: bool = False,
        global_gc_interval: int = 3 * 60 * 60,
        global_ttl: int = 3 * 60 * 60,
    ):
        """Create a file cache config.

        Args:
            chunk_size (int):
                Divide the token list into batches, each batch
                contains batchSize tokens. Defaults to 16.
            split_number (int):
                Split the hash value into the file with multiple directories.
                e.g, splitNumber=2, hash value=123456, the file path is 12/34/56.
            root (str):
                The root directory of the kv state files.
                Defaults to "/tmp/vineyard/llm_cache".
            filesystem_type (str):
                The type of the filesystem. Defaults to "local".
            client_gc_interval (int):
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
        self.split_number = split_number
        self.root = root
        self.filesystem_type = filesystem_type
        self.client_gc_interval = client_gc_interval
        self.ttl = ttl
        self.enable_global_gc = enable_global_gc
        self.global_gc_interval = global_gc_interval
        self.global_ttl = global_ttl
