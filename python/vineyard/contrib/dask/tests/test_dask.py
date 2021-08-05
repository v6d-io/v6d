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
import os
import pytest
import subprocess
import shutil
import sys
import time
import vineyard

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.dask.dask import register_dask_types

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from dask.distributed import Client


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_dask():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_dask_types(builder, resolver)
            yield builder, resolver


def find_executable(name):
    ''' Use executable in local build directory first.
    '''
    binary_dir = os.environ.get('DASK_EXECUTABLE_DIR', '')
    exe = os.path.join(binary_dir, name)
    if os.path.isfile(exe) and os.access(exe, os.R_OK):
        return exe
    exe = shutil.which(name)
    if exe is not None:
        return exe
    raise RuntimeError('Unable to find program %s' % name)


@contextlib.contextmanager
def start_program(name, *args, verbose=False, nowait=False, **kwargs):
    env, cmdargs = os.environ.copy(), list(args)
    for k, v in kwargs.items():
        if k[0].isupper():
            env[k] = str(v)
        else:
            cmdargs.append('--%s' % k)
            cmdargs.append(str(v))

    try:
        prog = find_executable(name)
        print('Starting %s...' % prog, flush=True)
        if verbose:
            out, err = sys.stdout, sys.stderr
        else:
            out, err = subprocess.PIPE, subprocess.PIPE
        proc = subprocess.Popen([prog] + cmdargs, env=env, stdout=out, stderr=err)
        if not nowait:
            time.sleep(1)
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError('Failed to launch program %s' % name)
        yield proc
    finally:
        print('Terminating %s' % prog, flush=True)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


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
def dask_info(vineyard_ipc_sockets):
    with launch_dask_cluster(vineyard_ipc_sockets, 'localhost', 8786) as r:
        yield r


def test_dask_array_builder(dask_info):
    clients, dask_scheduler, _ = dask_info
    arr = da.ones((1024, 1024), chunks=(256, 256))
    obj_id = clients[0].put(arr, dask_scheduler=dask_scheduler)
    meta = clients[0].get_meta(obj_id)
    assert meta['partitions_-size'] == 16


def test_dask_dataframe_builder(dask_info):
    clients, dask_scheduler, _ = dask_info
    arr = da.ones((1024, 2), chunks=(256, 2))
    df = dd.from_dask_array(arr, columns=['a', 'b'])
    obj_id = clients[0].put(df, dask_scheduler=dask_scheduler)
    meta = clients[0].get_meta(obj_id)
    assert meta['partitions_-size'] == 4


def test_dask_array_resolver(dask_info):
    clients, dask_scheduler, dask_workers = dask_info
    num = len(clients)

    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::GlobalTensor'
    meta['partitions_-size'] = num * num
    meta.set_global(True)

    for i in range(num):
        for j in range(num):
            obj = clients[(i + j) % num].put(np.array([i - j] * 8), partition_index=[i, j])
            clients[(i + j) % num].persist(obj)
            meta.add_member('partitions_-%d' % (i * num + j), obj)

    gtensor = clients[0].create_metadata(meta)
    clients[0].persist(gtensor)
    darr = clients[0].get(gtensor.id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    assert darr.sum().sum().compute() == 0


def test_dask_dataframe_resolver(dask_info):
    clients, dask_scheduler, dask_workers = dask_info
    num = len(clients)

    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::GlobalDataFrame'
    meta['partitions_-size'] = num
    meta.set_global(True)

    for i in range(num):
        obj = clients[i].put(pd.DataFrame({'x': [i, i * 2], 'y': [i * 3, i * 4]}))
        clients[i].persist(obj)
        meta.add_member('partitions_-%d' % i, obj)

    gdf = clients[0].create_metadata(meta)
    clients[0].persist(gdf)
    ddf = clients[0].get(gdf.id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    assert ddf.sum().sum().compute() == 60


def test_dask_array_roundtrip(dask_info):
    clients, dask_scheduler, dask_workers = dask_info
    arr = da.ones((1024, 1024), chunks=(256, 256))
    obj_id = clients[0].put(arr, dask_scheduler=dask_scheduler)
    arr1 = clients[0].get(obj_id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    np.testing.assert_allclose(arr1.compute(), np.ones((1024, 1024)))


def test_dask_dataframe_roundtrip(dask_info):
    clients, dask_scheduler, dask_workers = dask_info
    arr = da.ones((1024, 2), chunks=(256, 2))
    df = dd.from_dask_array(arr, columns=['a', 'b'])
    obj_id = clients[0].put(df, dask_scheduler=dask_scheduler)
    df1 = clients[0].get(obj_id, dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    pd.testing.assert_frame_equal(df1.compute(), pd.DataFrame({'a': np.ones(1024), 'b': np.ones(1024)}))