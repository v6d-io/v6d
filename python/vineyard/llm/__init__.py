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
        shape: Tuple[int] = None,
        dtype: str = None,
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
            shape (tuple, optional):
                The shape of kv tensor. Defaults to None.
                e,g, the shape of torch.rand(2, 2) is torch.Size([2, 2]).
            dtype (numpy.dtype, optional):
                The dtype of the tensor. Defaults to None.
                e.g., numpy.float32, numpy.float64.
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
        self.shape = shape
        # the dtype of the tensor
        self.tensor_dtype = dtype

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
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]
            kv_cache_list (List[Dict[int, Tuple[KVTensor, KVTensor]]]):
                the kv tensors list of the related tokens including all layers.

                The k, v tensor for i-th token at the j-th layer is: kv_state_list[i][j]
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
            kv_state_list: (List[Tuple[KVTensor, KVTensor]]):
                the kv tensors list of the related tokens including all layers.

                The k, v tensor for i-th token at the j-th layer is: kv_state_list[i][j]

        Returns:
            int: The number of matched tokens.
        """
        return self.kv_cache_manager.query(tokens, kv_state_list)

    def __del__(self):
        if self.kv_cache_manager:
            self.kv_cache_manager.close()
