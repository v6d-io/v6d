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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from .config import FileCacheConfig
from .config import VineyardCacheConfig
from .llm_C import KVTensor
from .llm_C import _generate


class KVCache:  # pylint: disable=too-many-instance-attributes
    """KVCache is a class that manages the llm kv cache in vineyard."""

    def __init__(
        self,
        cache_config: Union[VineyardCacheConfig, FileCacheConfig],
        tensor_bytes: int = 10,
        cache_capacity: int = 10,
        layer: int = 1,
        **kwargs
    ):
        """Create a llm kv cache manager based on vineyard blob.

        Args:
            cache_config (Union[VineyardCacheConfig, FileCacheConfig]):
                The config of the kv cache, including vineyard cache and file cache.
            tensor_bytes (int, optional):
                The size of the kv cache tensor.
                Defaults to 10.
            cache_capacity (int, optional):
                The capacity of the KV cache refers to the maximum number of
                tokens it can hold. Defaults to 10.
            layer (int, optional):
                The number of layers of the kv cache. Defaults to 1.
        """
        self.kv_cache_manager = None
        if not isinstance(cache_config, VineyardCacheConfig) and not isinstance(
            cache_config, FileCacheConfig
        ):
            raise ValueError(
                "The cache_config should be VineyardCacheConfig or FileCacheConfig."
            )
        self.tensor_bytes = tensor_bytes
        self.cache_capacity = cache_capacity
        self.layer = layer

        self.kv_cache_manager = _generate(
            tensor_bytes=tensor_bytes,
            cache_capacity=cache_capacity,
            layer=layer,
            **cache_config.__dict__,
            **kwargs
        )

    def update(
        self,
        prefix: Optional[List[int]],
        tokens: List[int],
        kv_state_list: List[List[Tuple[KVTensor, KVTensor]]],
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

                The k, v tensor for i-th token at the j-th layer is: kv_state_list[i][j]

                Whether the underlying kv cache is vineyard or file, the
                kv_state_list is managed by the caller.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_state_list as follows:

                .. code:: python

                    kv_state_list = []
                    for _ in range(2): # the number of tokens
                        k_tensor = np.random.rand(2,2).astype(np.float32)
                        v_tensor = np.random.rand(2,2).astype(np.float32)
                        kv_state_list.append(
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
            return self.kv_cache_manager.update(prefix, tokens, kv_state_list)
        else:
            return self.kv_cache_manager.update(tokens, kv_state_list)

    def query(
        self,
        tokens: List[int],
        kv_state_list: List[List[Tuple[KVTensor, KVTensor]]],
    ) -> int:
        """Query the kv cache stored in vineyard.

        Args:
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]
            kv_state_list: (List[List[Tuple[KVTensor, KVTensor]]]):
                the kv tensors list of the related tokens including all layers, and its
                length should be the same as the length of tokens.

                The k, v tensor for i-th token at the j-th layer is: kv_state_list[i][j]

                For VineyardConfigCache, the kv_state_list is managed by vineyard.
                The caller does not need to malloc and free the memory of the kv state.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_state_list as follows:

                .. code:: python

                    kv_state_list = [
                        (
                            KVTensor(0, 0),
                            KVTensor(0, 0),
                        ) for _ in range(2) # the number of layers
                    ] * 2 # the number of tokens

                For FileCacheConfig, the kv_state_list is managed by the caller.
                The caller needs to malloc and free the memory of the kv state.
                Assume the layer is 2, the tokens is [1, 2], then you should allocate
                the kv_state_list as follows:

                .. code:: python

                    kv_state_list = []
                    for _ in range(2): # the number of tokens
                        k_tensor = np.empty((2,2), dtype=np.float32)
                        v_tensor = np.empty((2,2), dtype=np.float32)
                        kv_state_list.append(
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
        return self.kv_cache_manager.query(tokens, kv_state_list)

    def __del__(self):
        if self.kv_cache_manager:
            self.kv_cache_manager.close()
