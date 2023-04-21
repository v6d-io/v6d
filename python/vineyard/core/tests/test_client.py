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
import multiprocessing
import random
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pytest

import vineyard
from vineyard._C import ObjectMeta
from vineyard.core import default_builder_context
from vineyard.core import default_resolver_context
from vineyard.data import register_builtin_types


def generate_vineyard_ipc_sockets(vineyard_ipc_sockets, nclients):
    return list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), nclients))


def generate_vineyard_ipc_clients(vineyard_ipc_sockets, nclients):
    vineyard_ipc_sockets = generate_vineyard_ipc_sockets(vineyard_ipc_sockets, nclients)
    return tuple(vineyard.connect(sock) for sock in vineyard_ipc_sockets)


register_builtin_types(default_builder_context, default_resolver_context)


def test_metadata(vineyard_client):
    xid = vineyard_client.put(1.2345)
    yid = vineyard_client.put(2.3456)
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', vineyard_client.get_meta(yid))
    meta.set_global(False)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)

    def go(meta):
        for k, v in meta.items():
            if isinstance(v, ObjectMeta):
                go(v)
            else:
                print('k-v in meta: ', k, v)

    meta = vineyard_client.get_meta(rmeta.id)
    go(meta)
    go(meta)
    go(meta)
    go(meta)


def test_metadata_global(vineyard_client):
    xid = vineyard_client.put(1.2345)
    yid = vineyard_client.put(2.3456)
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', vineyard_client.get_meta(yid))
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)

    def go(meta):
        for k, v in meta.items():
            if isinstance(v, ObjectMeta):
                go(v)
            else:
                print('k-v in meta: ', k, v)

    meta = vineyard_client.get_meta(rmeta.id)
    go(meta)
    go(meta)
    go(meta)
    go(meta)


def test_persist(vineyard_client):
    xid = vineyard_client.put(1.2345)
    yid = vineyard_client.put(2.3456)
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', yid)
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)


def test_persist_multiref(vineyard_client):
    xid = vineyard_client.put(1.2345)
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', xid)
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)


def test_concurrent_blob(vineyard_ipc_sockets):  # noqa: C901
    clients = generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    def job1(client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((1, 2, 3))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
        return True

    def job2(client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((2, 3, 4))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
        return True

    def job3(client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((3, 4, 5))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
        return True

    def job4(client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((4, 5, 6))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
        return True

    def job5(client):
        o = None
        try:
            o = client.get_object(client.put((np.ones((1, 2, 3)), np.ones((2, 3, 4)))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
        return True

    jobs = [job1, job2, job3, job4, job5]

    with ThreadPoolExecutor(32) as executor:
        fs, rs = [], []
        for _ in range(1024):
            job = random.choice(jobs)
            client = random.choice(clients)
            fs.append(executor.submit(job, client))
        for future in fs:
            rs.append(future.result())
        if not all(rs):
            pytest.fail("Failed to execute tests ...")


def test_concurrent_blob_mp(  # noqa: C901, pylint: disable=too-many-statements
    vineyard_ipc_sockets,
):
    num_proc = 32
    job_per_proc = 64

    vineyard_ipc_sockets = generate_vineyard_ipc_sockets(vineyard_ipc_sockets, num_proc)

    def job1(rs, state, client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((1, 2, 3))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            print('failed with %r: %s' % (o, e), flush=True)
            traceback.print_exc()
            state.value = -1
            rs.put((False, 'failed: %s' % e))
        else:
            rs.put((True, ''))

    def job2(rs, state, client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((2, 3, 4))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            print('failed with %r: %s' % (o, e), flush=True)
            traceback.print_exc()
            state.value = -1
            rs.put((False, 'failed: %s' % e))
        else:
            rs.put((True, ''))

    def job3(rs, state, client):
        o = None
        try:
            o = client.get_object(client.put(np.ones((3, 4, 5))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            print('failed with %r: %s' % (o, e), flush=True)
            traceback.print_exc()
            state.value = -1
            rs.put((False, 'failed: %s' % e))
        else:
            rs.put((True, ''))

    def job4(rs, state, client):
        o = None
        try:
            o = client.get_object(client.put((np.ones((4, 5, 6)))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            print('failed with %r: %s' % (o, e), flush=True)
            traceback.print_exc()
            state.value = -1
            rs.put((False, 'failed: %s' % e))
        else:
            rs.put((True, ''))

    def job5(rs, state, client):
        o = None
        try:
            o = client.get_object(client.put((np.ones((1, 2, 3)), np.ones((2, 3, 4)))))
            client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            print('failed with %r: %s' % (o, e), flush=True)
            traceback.print_exc()
            state.value = -1
            rs.put((False, 'failed: %s' % e))
        else:
            rs.put((True, ''))

    def start_requests(rs, state, ipc_socket):
        jobs = [job1, job2, job3, job4, job5]
        client = vineyard.connect(ipc_socket).fork()

        for _ in range(job_per_proc):
            if state.value != 0:
                break
            job = random.choice(jobs)
            job(rs, state, client)

    ctx = multiprocessing.get_context(method='fork')
    procs, rs, state = [], ctx.Queue(), ctx.Value('i', 0)
    for sock in vineyard_ipc_sockets:
        proc = ctx.Process(
            target=start_requests,
            args=(
                rs,
                state,
                sock,
            ),
        )
        proc.start()
        procs.append(proc)

    for _ in range(num_proc * job_per_proc):
        r, message = rs.get(block=True)
        if not r:
            pytest.fail(message)
