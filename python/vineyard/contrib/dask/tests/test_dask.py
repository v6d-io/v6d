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

import contextlib
import pytest

import numpy as np
import pandas as pd

import dask.array as da
import dask.dataframe as dd

import vineyard
from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.dask.dask import register_dask_types
from vineyard.data.dataframe import make_global_dataframe
from vineyard.data.tensor import make_global_tensor
from vineyard.deploy.utils import start_program


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_dask():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_dask_types(builder, resolver)
            yield builder, resolver


@contextlib.contextmanager
def launch_dask_cluster(vineyard_ipc_sockets, host, port):
    with contextlib.ExitStack() as stack:
        proc = start_program('dask-scheduler', '--host', host, '--port', str(port))
        stack.enter_context(proc)
        scheduler = f'tcp://{host}:{port}'
        clients = []
        workers = {}
        for sock in vineyard_ipc_sockets:
            client = vineyard.connect(sock)
            worker_name = 'dask_worker_%d' % client.instance_id
            workers[client.instance_id] = worker_name
            # launch a worker with corresponding name for each vineyard instance
            proc = start_program('dask-worker', scheduler, '--name', worker_name, VINEYARD_IPC_SOCKET=sock)
            stack.enter_context(proc)
            clients.append(client)
        yield clients, scheduler, workers


@pytest.fixture(scope="module", autouse=True)
def dask_cluster(vineyard_ipc_sockets):
    with launch_dask_cluster(vineyard_ipc_sockets, 'localhost', 8786) as cluster:
        yield cluster


def test_dask_array_builder(dask_cluster):
    clients, dask_scheduler, _ = dask_cluster
    arr = da.ones((1024, 1024), chunks=(256, 256))
    obj_id = clients[0].put(arr, dask_scheduler=dask_scheduler)
    meta = clients[0].get_meta(obj_id)
    assert meta['partitions_-size'] == 16


def test_dask_dataframe_builder(dask_cluster):
    clients, dask_scheduler, _ = dask_cluster
    arr = da.ones((1024, 2), chunks=(256, 2))
    df = dd.from_dask_array(arr, columns=['a', 'b'])
    obj_id = clients[0].put(df, dask_scheduler=dask_scheduler)
    meta = clients[0].get_meta(obj_id)
    assert meta['partitions_-size'] == 4


def test_dask_array_resolver(dask_cluster):
    clients, dask_scheduler, dask_workers = dask_cluster
    num = len(clients)

    chunks = []
    for i in range(num):
        for j in range(num):
            chunk = clients[(i + j) % num].put(np.array([i - j] * 8), partition_index=[i, j])
            clients[(i + j) % num].persist(chunk)
            chunks.append(chunk)

    gtensor = make_global_tensor(clients[0], chunks)
    darr = clients[0].get(gtensor.id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    assert darr.sum().sum().compute() == 0


def test_dask_dataframe_resolver(dask_cluster):
    clients, dask_scheduler, dask_workers = dask_cluster

    chunks = []
    for i, client in enumerate(clients):
        chunk = client.put(pd.DataFrame({'x': [i, i * 2], 'y': [i * 3, i * 4]}))
        client.persist(chunk)
        chunks.append(chunk)

    gdf = make_global_dataframe(clients[0], chunks)
    ddf = clients[0].get(gdf.id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    assert ddf.sum().sum().compute() == 60


def test_dask_array_roundtrip(dask_cluster):
    clients, dask_scheduler, dask_workers = dask_cluster
    arr = da.ones((1024, 1024), chunks=(256, 256))
    obj_id = clients[0].put(arr, dask_scheduler=dask_scheduler)
    arr1 = clients[0].get(obj_id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    np.testing.assert_allclose(arr1.compute(), np.ones((1024, 1024)))


def test_dask_dataframe_roundtrip(dask_cluster):
    clients, dask_scheduler, dask_workers = dask_cluster
    arr = da.ones((1024, 2), chunks=(256, 2))
    df = dd.from_dask_array(arr, columns=['a', 'b'])
    obj_id = clients[0].put(df, dask_scheduler=dask_scheduler)
    df1 = clients[0].get(obj_id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    pd.testing.assert_frame_equal(df1.compute(), pd.DataFrame({'a': np.ones(1024), 'b': np.ones(1024)}))
