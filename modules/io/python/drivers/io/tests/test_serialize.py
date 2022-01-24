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

import logging
import os
import shutil

import numpy as np

import pytest

import vineyard
import vineyard.io

logger = logging.getLogger('vineyard')


@pytest.fixture(scope='module')
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
    meta['typename'] = 'vineyard::Tuple'
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


def test_seriarialize_round_trip(vineyard_ipc_socket, vineyard_endpoint, global_object):
    destination = '/tmp/seri-test'
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


@pytest.mark.skip("require oss")
def test_seriarialize_round_trip_on_oss(
    vineyard_ipc_socket, vineyard_endpoint, global_object
):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")
    vineyard.io.serialize(
        'oss://grape-uk/tmp/seri-test',
        global_object,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "endpoint": endpoint,
        },
    )
    ret = vineyard.io.deserialize(
        'oss://grape-uk/tmp/seri-test',
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "endpoint": endpoint,
        },
    )

    client = vineyard.connect(vineyard_ipc_socket)
    expected = client.get(global_object)
    actual = client.get(ret)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)

    assert len(expected) == len(actual)

    for item1, item2 in zip(expected, actual):
        np.testing.assert_array_almost_equal(item1, item2)


@pytest.mark.skip(reason="require s3")
def test_seriarialize_round_trip_on_s3(
    vineyard_ipc_socket, vineyard_endpoint, global_object
):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    region_name = os.environ.get("REGION", "us-east-1")
    vineyard.io.serialize(
        "s3://test-bucket/tmp/seri-test",
        global_object,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {"region_name": region_name},
        },
    )
    ret = vineyard.io.deserialize(
        's3://test-bucket/tmp/seri-test',
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {"region_name": region_name},
        },
    )

    client = vineyard.connect(vineyard_ipc_socket)
    expected = client.get(global_object)
    actual = client.get(ret)

    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)

    assert len(expected) == len(actual)

    for item1, item2 in zip(expected, actual):
        np.testing.assert_array_almost_equal(item1, item2)
