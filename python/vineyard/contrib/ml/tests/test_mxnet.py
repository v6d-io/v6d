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

import mxnet as mx

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.ml.mxnet import register_mxnet_types


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_mxnet():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_mxnet_types(builder, resolver)
            yield builder, resolver


def test_mx_tensor(vineyard_client):
    data = [np.random.rand(2, 3) for i in range(10)]
    label = [np.random.rand(2, 3) for i in range(10)]
    dataset = mx.gluon.data.ArrayDataset((data, label))
    object_id = vineyard_client.put(dataset, typename='Tensor')
    dtrain = vineyard_client.get(object_id)
    assert len(dataset[0]) == len(dtrain[0])
    assert dataset[0][0].shape == dtrain[0][0].shape
    assert dataset[1][0].shape == dtrain[1][0].shape


def test_mx_dataframe(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
    label = df['c'].values.astype(np.float32)
    data = df.drop('c', axis=1).values.astype(np.float32)
    dataset = mx.gluon.data.ArrayDataset((data, label))
    object_id = vineyard_client.put(dataset, typename="DataFrame", cols=['a', 'b', 'c'], label='c')
    dtrain = vineyard_client.get(object_id, label='c')
    assert len(dtrain[0]) == 4
    assert dataset[0].shape == dtrain[0].shape
    assert dataset[1].shape == dtrain[1].shape


def test_mx_record_batch(vineyard_client):
    arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
    object_id = vineyard_client.put(batch)
    dtrain = vineyard_client.get(object_id, label='target')
    assert len(dtrain[0]) == 4
    assert len(dtrain[0][0]) == 2


def test_mx_table(vineyard_client):
    arrays = [pa.array([1, 2]), pa.array([0, 1]), pa.array([0.1, 0.2])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
    batches = [batch] * 4
    table = pa.Table.from_batches(batches)
    object_id = vineyard_client.put(table)
    dtrain = vineyard_client.get(object_id, label='target')
    assert len(dtrain[0]) == 8
    assert len(dtrain[0][0]) == 2
