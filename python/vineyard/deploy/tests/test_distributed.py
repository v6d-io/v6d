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

import itertools
import json
import logging
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import numpy as np

import vineyard
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)

logger = logging.getLogger('vineyard')


def generate_vineyard_ipc_clients(vineyard_ipc_sockets, nclients):
    vineyard_ipc_sockets = list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), nclients))
    return tuple(vineyard.connect(sock) for sock in vineyard_ipc_sockets)


def test_get_after_persist(vineyard_ipc_sockets):
    client1, client2 = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 2)

    data = np.ones((1, 2, 3, 4, 5))
    o = client1.put(data)
    client1.persist(o)
    meta = client2.get_meta(o, True)
    assert data.shape == tuple(json.loads(meta['shape_']))


def test_add_remote_placeholder(vineyard_ipc_sockets):
    client1, client2, client3, client4 = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    data = np.ones((1, 2, 3, 4, 5))

    o1 = client1.put(data)
    o2 = client2.put(data)
    o3 = client3.put(data)
    o4 = client4.put(data)

    client4.persist(o4)
    client3.persist(o3)
    client2.persist(o2)
    client1.persist(o1)

    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::Tuple'
    meta['size_'] = 4
    meta.set_global(True)
    meta.add_member('__elements_-0', o1)
    meta.add_member('__elements_-1', o2)
    meta.add_member('__elements_-2', o3)
    meta.add_member('__elements_-3', o4)
    meta['__elements_-size'] = 4
    tupid = client1.create_metadata(meta)
    client1.persist(tupid)

    meta = client2.get_meta(tupid, True)
    assert meta['__elements_-size'] == 4


def test_add_remote_placeholder_with_sync(vineyard_ipc_sockets):
    client1, client2, client3, client4 = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    data = np.ones((1, 2, 3, 4, 5))

    o1 = client1.put(data)
    client1.persist(o1)
    time.sleep(20)

    o2 = client2.put(data)
    client2.persist(o2)
    time.sleep(20)

    o3 = client3.put(data)
    client3.persist(o3)
    time.sleep(20)

    o4 = client4.put(data)
    client4.persist(o4)
    time.sleep(20)

    client1.get_meta(o4)
    client2.get_meta(o1)
    client3.get_meta(o2)
    client4.get_meta(o3)


def test_remote_deletion(vineyard_ipc_sockets):
    client1, client2 = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 2)

    client1 = vineyard.connect(vineyard_ipc_sockets[0])
    client2 = vineyard.connect(vineyard_ipc_sockets[1])

    old_status = client1.status

    data = np.ones((1, 2, 3, 4, 5))
    o1 = client1.put(data)
    client1.persist(o1)

    new_status = client1.status

    assert old_status.memory_limit == new_status.memory_limit
    assert old_status.memory_usage != new_status.memory_usage

    client2.get_meta(o1, sync_remote=True)
    client2.delete(o1)
    client1.sync_meta()

    new_status = client1.status

    assert old_status.memory_limit == new_status.memory_limit
    assert old_status.memory_usage == new_status.memory_usage


def test_concurrent_blob(vineyard_ipc_sockets):
    client1, client2, client3, client4 = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    # FIXME: test concurrent blob creation and destory
    print(client1)
    print(client2)
    print(client3)
    print(client4)


def test_concurrent_meta(vineyard_ipc_sockets):
    clients = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    def job1(client):
        try:
            o = client.get_object(client.put(1))
            if random.random() > 0.5:
                client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job2(client):
        try:
            o = client.get_object(client.put(1.23456))
            if random.random() > 0.5:
                client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job3(client):
        try:
            o = client.get_object(client.put('xxxxabcd'))
            if random.random() > 0.5:
                client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job4(client):
        try:
            o = client.get_object(client.put((1, 1.2345)))
            if random.random() > 0.5:
                client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job5(client):
        try:
            o = client.get_object(client.put((1, 1.2345, 'xxxxabcd')))
            if random.random() > 0.5:
                client.delete(o.id)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
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


def test_concurrent_persist(vineyard_ipc_sockets):
    clients = \
            generate_vineyard_ipc_clients(vineyard_ipc_sockets, 4)

    def job1(client):
        try:
            o = client.get_object(client.put(1))
            client.persist(o.id)
            if random.random() > 0.5:
                client.delete(o.id)
            else:
                client.sync_meta()
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job2(client):
        try:
            o = client.get_object(client.put(1.23456))
            client.persist(o.id)
            if random.random() > 0.5:
                client.delete(o.id)
            else:
                client.sync_meta()
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job3(client):
        try:
            o = client.get_object(client.put('xxxxabcd'))
            client.persist(o.id)
            if random.random() > 0.5:
                client.delete(o.id)
            else:
                client.sync_meta()
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job4(client):
        try:
            o = client.get_object(client.put((1, 1.2345)))
            client.persist(o.id)
            if random.random() > 0.5:
                client.delete(o.id)
            else:
                client.sync_meta()
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    def job5(client):
        try:
            o = client.get_object(client.put((1, 1.2345, 'xxxxabcd')))
            client.persist(o.id)
            if random.random() > 0.5:
                client.delete(o.id)
            else:
                client.sync_meta()
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail('failed: %s' % e)
            return False
        return True

    jobs = [job1, job2, job3, job4, job5]

    with ThreadPoolExecutor(16) as executor:
        fs, rs = [], []
        for _ in range(256):
            job = random.choice(jobs)
            client = random.choice(clients)
            fs.append(executor.submit(job, client))
        for future in fs:
            rs.append(future.result())
        if not all(rs):
            pytest.fail("Failed to execute tests ...")
