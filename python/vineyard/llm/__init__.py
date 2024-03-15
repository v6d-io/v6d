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

import numpy as np

import torch
from torch import dtype

import vineyard

from .llm_C import KVTensor
from .llm_C import _generate


class KV_Cache:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        socket: str,
        tensor_bytes: int = 10,
        cache_capacity: int = 10,
        layer: int = 1,
        torch_size: torch.Size = None,
        dtype: dtype = None,
        block_size: int = 5,
        sync_interval: int = 3,
        llm_cache_sync_lock: str = "llmCacheSyncLock",
        llm_cache_object_name: str = "llm_cache_object",
        llm_ref_cnt_object_name: str = "llm_refcnt_object",
        **kwargs
    ):
        self.client = vineyard.connect(socket)
        self.tensor_bytes = tensor_bytes
        self.cache_capacity = cache_capacity
        self.layer = layer
        self.torch_size = torch_size
        # the dtype of the tensor
        self.tensor_dtype = dtype
        # the dtype of the numpy array of the tensor
        self.numpy_dtype = None
        self.block_size = block_size
        self.sync_interval = sync_interval
        self.llm_cache_sync_lock = llm_cache_sync_lock
        self.llm_cache_object_name = llm_cache_object_name
        self.llm_ref_cnt_object_name = llm_ref_cnt_object_name
        self.kv_cache_manager = _generate(
            ipc_client=self.client.ipc_client,
            tensor_bytes=tensor_bytes,
            cache_capacity=cache_capacity,
            layer=layer,
            block_size=block_size,
            sync_interval=sync_interval,
            llm_cache_sync_lock=llm_cache_sync_lock,
            llm_cache_object_name=llm_cache_object_name,
            llm_ref_cnt_object_name=llm_ref_cnt_object_name,
            **kwargs
        )

    def update(
        self,
        tokens: list,
        kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
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
