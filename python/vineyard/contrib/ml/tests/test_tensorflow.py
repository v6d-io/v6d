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

import tensorflow as tf

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.ml.tensorflow import register_tf_types


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_tensorflow():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_tf_types(builder, resolver)
            yield builder, resolver


def test_tf_tensor(vineyard_client):
    data = [np.random.rand(2, 3) for i in range(10)]
    label = [np.random.rand(2, 3) for i in range(10)]
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    object_id = vineyard_client.put(dataset)
    dtrain = vineyard_client.get(object_id)
    for x, y in dataset.take(1):
        xdata = x.shape
        ydata = y.shape
    for x, y in dtrain.take(1):
        xdtrain = x.shape
        ydtrain = y.shape
    assert xdata == xdtrain
    assert ydata == ydtrain
    assert len(dataset) == len(dtrain)


def test_tf_dataframe(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'target': [1.0, 2.0, 3.0, 4.0]})
    labels = df.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    object_id = vineyard_client.put(dataset)
    dtrain = vineyard_client.get(object_id)
    for x, y in dataset.take(1):
        data_ncols = len(list(x.keys()))
    for x, y in dtrain.take(1):
        dtrain_ncols = len(list(x.keys()))
    assert len(dataset) == len(dtrain)
    assert data_ncols == dtrain_ncols


def test_tf_record_batch(vineyard_client):
    arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
    object_id = vineyard_client.put(batch)
    dtrain = vineyard_client.get(object_id)
    for x, y in dtrain.take(1):
        ncols = len(list(x.keys()))
    assert ncols == 2
    assert len(dtrain) == 4


def test_tf_table(vineyard_client):
    arrays = [pa.array([1, 2]), pa.array([0, 1]), pa.array([0.1, 0.2])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'label'])
    batches = [batch] * 4
    table = pa.Table.from_batches(batches)
    object_id = vineyard_client.put(table)
    dtrain = vineyard_client.get(object_id)
    for x, y in dtrain.take(1):
        ncols = len(list(x.keys()))
    assert ncols == 2
    assert len(dtrain) == 8
