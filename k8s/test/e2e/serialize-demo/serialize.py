#! /usr/bin/env python3
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

import time
import logging
import shutil

import numpy as np

import vineyard
import vineyard.io

logger = logging.getLogger('vineyard')

def global_object(vineyard_ipc_socket):
    client1 = vineyard.connect(vineyard_ipc_socket)
    client2 = vineyard.connect(vineyard_ipc_socket)
    client3 = vineyard.connect(vineyard_ipc_socket)
    client4 = vineyard.connect(vineyard_ipc_socket)

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
    meta['typename'] = 'vineyard::Sequence'
    meta['size_'] = 4
    meta.set_global(True)
    meta.add_member('__elements_-0', client1.get_meta(o1))
    meta.add_member('__elements_-1', client1.get_meta(o2))
    meta.add_member('__elements_-2', o3)
    meta.add_member('__elements_-3', o4)
    meta['__elements_-size'] = 4
    tup = client1.create_metadata(meta)
    client1.persist(tup)
    return tup.id

def test_seriarialize_round_trip(destination, vineyard_ipc_socket, vineyard_endpoint, global_object):
    shutil.rmtree(destination, ignore_errors=True)
    vineyard.io.serialize(
        destination,
        global_object,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    logger.info("finish serializing object to %s", destination)
    ret = vineyard.io.deserialize(
        destination,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    logger.info("finish deserializing object from %s, as %s", destination, ret)

    client = vineyard.connect(vineyard_ipc_socket)
    expected = client.get(global_object)
    actual = client.get(ret)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)

    assert len(expected) == len(actual)

    for item1, item2 in zip(expected, actual):
        np.testing.assert_array_almost_equal(item1, item2)

    return "test passed"

destination = '/var/vineyard/serialize'
vineyard_ipc_socket = '/var/run/vineyard.sock'
svc = 'vineyardd-sample-rpc.vineyard-system'
vineyard_endpoint = (svc, 9600)
globalobject = global_object(vineyard_ipc_socket)
res = test_seriarialize_round_trip(destination, vineyard_ipc_socket, vineyard_endpoint, globalobject)
print(res,flush=True)

# avoid CrashLoopBackOff
time.sleep(600)
