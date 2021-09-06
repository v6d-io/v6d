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

import os
import pytest
import numpy as np

import vineyard
import vineyard.io


@pytest.fixture(scope='module')
def global_obj(vineyard_ipc_socket):
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
    meta.add_member('__elements_-0', o1)
    meta.add_member('__elements_-1', o2)
    meta.add_member('__elements_-2', o3)
    meta.add_member('__elements_-3', o4)
    meta['__elements_-size'] = 4
    tup = client1.create_metadata(meta)
    client1.persist(tup)
    return tup.id


def test_seriarialize_round_trip(vineyard_ipc_socket, vineyard_endpoint, global_obj):
    vineyard.io.serialize('/tmp/seri-test',
                          global_obj,
                          vineyard_ipc_socket=vineyard_ipc_socket,
                          vineyard_endpoint=vineyard_endpoint)
    ret = vineyard.io.deserialize('/tmp/seri-test',
                                  vineyard_ipc_socket=vineyard_ipc_socket,
                                  vineyard_endpoint=vineyard_endpoint)
    client = vineyard.connect(vineyard_ipc_socket)
    old_meta = client.get_meta(global_obj)
    new_meta = client.get_meta(ret)
    print('old meta', old_meta)
    print('new meta', new_meta)


@pytest.mark.skip("require oss")
def test_seriarialize_round_trip_on_oss(vineyard_ipc_socket, vineyard_endpoint, global_obj):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")
    vineyard.io.serialize('oss://grape-uk/tmp/seri-test',
                          global_obj,
                          vineyard_ipc_socket=vineyard_ipc_socket,
                          vineyard_endpoint=vineyard_endpoint,
                          storage_options={
                              "key": accessKeyID,
                              "secret": accessKeySecret,
                              "endpoint": endpoint,
                          })
    ret = vineyard.io.deserialize('oss://grape-uk/tmp/seri-test',
                                  vineyard_ipc_socket=vineyard_ipc_socket,
                                  vineyard_endpoint=vineyard_endpoint,
                                  storage_options={
                                      "key": accessKeyID,
                                      "secret": accessKeySecret,
                                      "endpoint": endpoint,
                                  })
    client = vineyard.connect(vineyard_ipc_socket)
    old_meta = client.get_meta(global_obj)
    new_meta = client.get_meta(ret)
    print('old meta', old_meta)
    print('new meta', new_meta)


@pytest.mark.skip(reason="require s3")
def test_seriarialize_round_trip_on_s3(vineyard_ipc_socket, vineyard_endpoint, global_obj):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    region_name = os.environ.get("REGION", "us-east-1")
    vineyard.io.serialize(
        "s3://test-bucket/tmp/seri-test",
        global_obj,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {
                "region_name": region_name
            },
        },
    )
    ret = vineyard.io.deserialize(
        's3://test-bucket/tmp/seri-test',
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {
                "region_name": region_name
            },
        },
    )
    client = vineyard.connect(vineyard_ipc_socket)
    old_meta = client.get_meta(global_obj)
    new_meta = client.get_meta(ret)
    print('old meta', old_meta)
    print('new meta', new_meta)
