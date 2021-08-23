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

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.ml.xgboost import register_xgb_types


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_xgboost():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_xgb_types(builder, resolver)
            yield builder, resolver


def test_numpy_ndarray(vineyard_client):
    arr = np.random.rand(4, 5)
    object_id = vineyard_client.put(arr)
    dtrain = vineyard_client.get(object_id)
    assert dtrain.num_col() == 5
    assert dtrain.num_row() == 4


def test_pandas_dataframe_specify_label(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [1.0, 2.0, 3.0, 4.0]})
    object_id = vineyard_client.put(df)
    dtrain = vineyard_client.get(object_id, label='a')
    assert dtrain.num_col() == 2
    assert dtrain.num_row() == 4
    arr = np.array([1, 2, 3, 4])
    assert np.allclose(arr, dtrain.get_label())
    assert dtrain.feature_names == ['b', 'c']


def test_pandas_dataframe_specify_data(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [[5, 1.0], [6, 2.0], [7, 3.0], [8, 9.0]]})
    object_id = vineyard_client.put(df)
    dtrain = vineyard_client.get(object_id, data='b', label='a')
    assert dtrain.num_col() == 2
    assert dtrain.num_row() == 4
    arr = np.array([1, 2, 3, 4])
    assert np.allclose(arr, dtrain.get_label())


def test_record_batch_xgb_resolver(vineyard_client):
    arrays = [pa.array([1, 2, 3, 4]), pa.array([3.0, 4.0, 5.0, 6.0]), pa.array([0, 1, 0, 1])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'f1', 'target'])
    object_id = vineyard_client.put(batch)
    dtrain = vineyard_client.get(object_id, label='target')
    assert dtrain.num_col() == 2
    assert dtrain.num_row() == 4
    arr = np.array([0, 1, 0, 1])
    assert np.allclose(arr, dtrain.get_label())
    assert dtrain.feature_names == ['f0', 'f1']


def test_table_xgb_resolver(vineyard_client):
    arrays = [pa.array([1, 2]), pa.array([0, 1]), pa.array([0.1, 0.2])]
    batch = pa.RecordBatch.from_arrays(arrays, ['f0', 'label', 'f2'])
    batches = [batch] * 3
    table = pa.Table.from_batches(batches)
    object_id = vineyard_client.put(table)
    dtrain = vineyard_client.get(object_id, label='label')
    assert dtrain.num_col() == 2
    assert dtrain.num_row() == 6
    arr = np.array([0, 1, 0, 1, 0, 1])
    assert np.allclose(arr, dtrain.get_label())
    assert dtrain.feature_names == ['f0', 'f2']
