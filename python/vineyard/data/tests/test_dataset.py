#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
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

from torch.utils.data import Dataset
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


class RandomDataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.ds = [(np.random.rand(4, 5, 6), np.random.rand(2, 3)) for i in range(num)]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.ds[idx]

def test_dataset(vineyard_client):
    ds = RandomDataset(10)
    object_id = vineyard_client.put(ds)
    new_ds = vineyard_client.get(object_id)

    for i in range(len(ds)):
        np.testing.assert_allclose(ds[i][0], new_ds[i][0])
        np.testing.assert_allclose(ds[i][1], new_ds[i][1])
