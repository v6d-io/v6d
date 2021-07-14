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

import pandas as pd
import pytest
import numpy as np

from vineyard.data.pickle import PickledReader, PickledWriter

b1m = 1 * 1024 * 1024
b16m = 16 * 1024 * 1024
b64m = 64 * 1024 * 1024
b128m = 128 * 1024 * 1024

values = [
    (b1m, 1),
    (b1m, True),
    (b1m, False),
    (b1m, (True, False)),
    (b1m, [True, False]),
    (b1m, (1, 2, 3)),
    (b1m, [1, 2, 3, 4]),
    (b1m, "dsdfsdf"),
    (b1m, (1, "sdfsdfs")),
    (b1m, b"dsdfsdf"),
    (b1m, memoryview(b"sdfsdfs")),
    (b1m, [1] * 100000000),
    (b1m, np.bool_(True)),
    (b1m, np.bool_(False)),
    (b1m, (np.bool_(True), np.bool_(False))),
    (b1m, [np.bool_(True), np.bool_(False)]),
    (b1m, np.arange(1024 * 1024 * 400)),
    (b16m, np.zeros((1024, 1024, 48), dtype='bool')),
    (b16m, np.zeros((1024, 1024, 48))),
    (b64m, np.zeros((1024, 1024, 512))),
    (b1m, pd.DataFrame({
        'a': np.ones(1024),
        'b': np.zeros(1024),
    })),
    (b16m, pd.DataFrame({
        'a': np.ones(1024 * 1024),
        'b': np.zeros(1024 * 1024),
    })),
    (b64m, pd.DataFrame({
        'a': np.ones(1024 * 1024 * 4),
        'b': np.zeros(1024 * 1024 * 4),
    })),
    (b128m, pd.DataFrame({
        'a': np.ones(1024 * 1024 * 16),
        'b': np.zeros(1024 * 1024 * 16),
    })),
]


def read_and_build(block_size, value):
    reader = PickledReader(value)
    bs, nlen = [], 0
    while True:
        block = reader.read(block_size)
        if block:
            bs.append(block)
            nlen += len(block)
        else:
            break
    assert nlen == reader.store_size

    writer = PickledWriter(reader.store_size)
    for block in bs:
        writer.write(block)
    assert writer.value is None
    writer.close()
    assert writer.value is not None
    return writer.value


@pytest.mark.parametrize("block_size, value", values)
def test_bytes_io_roundtrip(block_size, value):
    target = read_and_build(block_size, value)

    # compare values
    if isinstance(value, np.ndarray):
        # FIXME why `assert_array_equal` are so slow ...
        #
        # np.testing.assert_array_equal(target, value)
        #
        assert (target == value).all()
    elif isinstance(value, pd.DataFrame):
        pd.testing.assert_frame_equal(target, value)
    elif isinstance(value, pd.Index):
        pd.testing.assert_index_equal(target, value)
    elif isinstance(value, pd.Series):
        pd.testing.assert_series_equal(target, value)
    else:
        assert target == value


def test_bytes_io_numpy_ndarray(vineyard_client):
    arr = np.random.rand(4, 5, 6)
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)


def test_bytes_io_empty_ndarray(vineyard_client):
    arr = np.ones(())
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.ones((0, 1))
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.ones((0, 1, 2))
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.ones((0, 1, 2, 3))
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.zeros((), dtype='int')
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.zeros((0, 1), dtype='int')
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.zeros((0, 1, 2), dtype='int')
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)

    arr = np.zeros((0, 1, 2, 3), dtype='int')
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_allclose(arr, target)


def test_bytes_io_str_ndarray(vineyard_client):
    arr = np.array(['', 'x', 'yz', 'uvw'])
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_equal(arr, target)


def test_object_ndarray(vineyard_client):
    arr = np.array([1, 'x', 3.14, (1, 4)], dtype=object)
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_equal(arr, target)

    arr = np.ones((), dtype='object')
    object_id = vineyard_client.put(arr)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    np.testing.assert_equal(arr, target)


def test_bytes_io_tensor_order(vineyard_client):
    arr = np.asfortranarray(np.random.rand(10, 7))
    object_id = vineyard_client.put(arr)
    res = read_and_build(b1m, vineyard_client.get(object_id))
    assert res.flags['C_CONTIGUOUS'] == arr.flags['C_CONTIGUOUS']
    assert res.flags['F_CONTIGUOUS'] == arr.flags['F_CONTIGUOUS']


def test_bytes_io_pandas_dataframe(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
    object_id = vineyard_client.put(df)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    pd.testing.assert_frame_equal(df, target)


def test_bytes_io_pandas_dataframe_int_columns(vineyard_client):
    df = pd.DataFrame({1: [1, 2, 3, 4], 2: [5, 6, 7, 8]})
    object_id = vineyard_client.put(df)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    pd.testing.assert_frame_equal(df, target)


def test_bytes_io_pandas_dataframe_mixed_columns(vineyard_client):
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 1: [9, 10, 11, 12], 2: [13, 14, 15, 16]})
    object_id = vineyard_client.put(df)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    pd.testing.assert_frame_equal(df, target)


def test_bytes_io_pandas_series(vineyard_client):
    s = pd.Series([1, 3, 5, np.nan, 6, 8], name='foo')
    object_id = vineyard_client.put(s)
    target = read_and_build(b1m, vineyard_client.get(object_id))
    pd.testing.assert_series_equal(s, target)
