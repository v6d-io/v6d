#! /usr/bin/env python
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

import numpy as np
import pandas as pd

import pyarrow as pa
import pytest

import torch
from torch.utils.data import Dataset

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.ml.pytorch import register_torch_types


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_pytorch():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_torch_types(builder, resolver)
            yield builder, resolver


class TestData(Dataset):
    def __init__(self, num):
        self.num = num
        self.ds = [(np.random.rand(2, 3), np.random.rand(2, 3)) for i in range(num)]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.ds[idx]


def test_torch_tensor(vineyard_client):
    dataset = TestData(5)
    object_id = vineyard_client.put(dataset, typename='Tensor')
    dtrain = vineyard_client.get(object_id)
    xdata, ydata = dataset[0]
    xdtrain, ydtrain = dtrain[0]
    assert xdata.shape == xdtrain.shape
    assert ydata.shape == ydtrain.shape


def test_torch_dataframe(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
    label = torch.tensor(df['c'].values.astype(np.float32))
    data = torch.tensor(df.drop('c', axis=1).values.astype(np.float32))
    dataset = torch.utils.data.TensorDataset(data, label)
    object_id = vineyard_client.put(dataset, typename='Dataframe', cols=['a', 'b', 'c'], label='c')
    dtrain = vineyard_client.get(object_id, label='c')
    assert len(dtrain) == 4
    assert list(dtrain[0][0].size())[0] == 2


def test_tf_record_batch(vineyard_client):
    arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
    object_id = vineyard_client.put(batch)
    dtrain = vineyard_client.get(object_id, label='target')
    assert len(dtrain) == 4
    assert list(dtrain[0][0].size())[0] == 2


def test_tf_table(vineyard_client):
    arrays = [pa.array([1, 2]), pa.array([0, 1]), pa.array([0.1, 0.2])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
    batches = [batch] * 4
    table = pa.Table.from_batches(batches)
    object_id = vineyard_client.put(table)
    dtrain = vineyard_client.get(object_id, label='target')
    assert len(dtrain) == 8
    assert list(dtrain[0][0].size())[0] == 2
