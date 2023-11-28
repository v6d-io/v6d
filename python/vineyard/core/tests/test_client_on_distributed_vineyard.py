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

import itertools

import numpy as np
import pandas as pd
import pyarrow as pa

import pytest

import vineyard


def generate_vineyard_ipc_sockets(vineyard_ipc_sockets, nclients):
    return list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), nclients))


def generate_vineyard_ipc_clients(vineyard_ipc_sockets, nclients):
    vineyard_ipc_sockets = generate_vineyard_ipc_sockets(vineyard_ipc_sockets, nclients)
    return tuple(vineyard.connect(sock) for sock in vineyard_ipc_sockets)


@pytest.mark.parametrize(
    "value",
    [
        1,
        'abcde',
        True,
        (1, "2", pytest.approx(3.456), 4444, "5.5.5.5.5.5.5"),
        {1: 2, 3: 4, 5: None, None: 6},
        np.asfortranarray(np.random.rand(10, 7)),
        np.zeros((0, 1, 2, 3), dtype='int'),
        pa.array([1, 2, None, 3]),
        pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]}),
        pd.Series([1, 3, 5, np.nan, 6, 8], name='foo'),
    ],
)
def test_get_and_put_with_different_vineyard_instances(
    value, vineyard_rpc_client, vineyard_ipc_sockets
):
    ipc_clients = generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)
    objects = []

    if isinstance(value, pd.arrays.SparseArray):
        value = pd.DataFrame(value)

    for client in ipc_clients:
        o = client.put(value, persist=True)
        objects.append(o)
    o = vineyard_rpc_client.put(value, persist=True)
    objects.append(o)

    values = []
    for o in objects:
        for client in ipc_clients:
            values.append(client.get(vineyard.ObjectID(o)))
        values.append(vineyard_rpc_client.get(vineyard.ObjectID(o)))

    for v in values:
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, v)
        elif isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, v)
        else:
            value == v
