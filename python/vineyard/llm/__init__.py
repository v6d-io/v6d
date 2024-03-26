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

from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import torch
from torch import dtype

from .config import FileCacheConfig
from .config import VineyardCacheConfig
from .llm_C import KVTensor
from .llm_C import _generate


class KV_Cache:  # pylint: disable=too-many-instance-attributes
    """KV_Cache is a class that manages the llm kv cache in vineyard."""

    def __init__(
        self,
        cache_config: Union[VineyardCacheConfig, FileCacheConfig],
        tensor_bytes: int = 10,
        cache_capacity: int = 10,
        layer: int = 1,
        torch_size: torch.Size = None,
        dtype: dtype = None,
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
            torch_size (torch.Size, optional):
                The size of kv tensor. Defaults to None.
                e,g, the size of torch.rand(2, 2) is torch.Size([2, 2]).
            dtype (dtype, optional):
                The dtype of the tensor. Defaults to None.
                e.g., torch.float32, torch.float64.
        """
        if not isinstance(cache_config, VineyardCacheConfig) and not isinstance(
            cache_config, FileCacheConfig
        ):
            raise ValueError(
                "The cache_config should be VineyardCacheConfig or FileCacheConfig."
            )
        self.tensor_bytes = tensor_bytes
        self.cache_capacity = cache_capacity
        self.layer = layer
        self.torch_size = torch_size
        # the dtype of the tensor
        self.tensor_dtype = dtype
        # the dtype of the numpy array of the tensor
        self.numpy_dtype = None

        self.kv_cache_manager = _generate(
            tensor_bytes=tensor_bytes,
            cache_capacity=cache_capacity,
            layer=layer,
            **cache_config.__dict__,
            **kwargs
        )

    def update(
        self,
        tokens: list,
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """Update the kv cache stored in vineyard.

        Args:
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]
            kv_cache_list (List[Tuple[torch.Tensor, torch.Tensor]]):
                the kv tensors list of the related tokens including all layers.
                if the layer is 2, the kv_cache_list should be like:

        .. code:: bash

            [(k1, v1)[layer0], (k1, v1)[layer1],
             (k2, v2)[layer0], (k2, v2)[layer1],
             (k3, v3)[layer0], (k3, v3)[layer1],
             (k4, v4)[layer0], (k4, v4)[layer1]]

        """
        kv_state_list = []
        kv_state_entry = {}
        j = 0
        for k_tensor, v_tensor in kv_cache_list:
            k_tensor_numpy = k_tensor.numpy()
            v_tensor_numpy = v_tensor.numpy()
            if self.numpy_dtype is None:
                self.numpy_dtype = np.dtype(k_tensor_numpy.dtype)
            k_tensor_in_vineyard = KVTensor(k_tensor_numpy.data, k_tensor_numpy.nbytes)
            v_tensor_in_vineyard = KVTensor(v_tensor_numpy.data, v_tensor_numpy.nbytes)
            kv_state_entry[j] = (k_tensor_in_vineyard, v_tensor_in_vineyard)
            j += 1
            if j == self.layer:
                j = 0
                kv_state_list.append(kv_state_entry)
                kv_state_entry = {}
        self.kv_cache_manager.update(tokens, kv_state_list)

    def query(
        self,
        tokens: list,
    ):
        """Query the kv cache stored in vineyard.

        Args:
            tokens (list): the tokens of the kv cache
                e,g, [1 2 3 4]

        Returns:
            (List[Tuple[torch.Tensor, torch.Tensor]]):
                the kv tensors list of the related tokens including all layers.
                if the layer is 2, the kv_cache_list should be like:

            .. code:: bash

                [(k1, v1)[layer0], (k1, v1)[layer1],
                 (k2, v2)[layer0], (k2, v2)[layer1],
                 (k3, v3)[layer0], (k3, v3)[layer1],
                 (k4, v4)[layer0], (k4, v4)[layer1]]

        """
        kv_state_list = []
        kv_cache_list = []
        self.kv_cache_manager.query(tokens, kv_state_list)
        for k_state, v_state in kv_state_list:
            k_tensor_numpy = np.frombuffer(
                k_state.data, dtype=self.numpy_dtype
            ).reshape(self.torch_size)
            v_tensor_numpy = np.frombuffer(
                v_state.data, dtype=self.numpy_dtype
            ).reshape(self.torch_size)
            k_tensor = torch.from_numpy(k_tensor_numpy)
            v_tensor = torch.from_numpy(v_tensor_numpy)
            kv_cache_list.append((k_tensor, v_tensor))
        return kv_cache_list

    def __del__(self):
        self.kv_cache_manager.close()
