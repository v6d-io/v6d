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

import vineyard
import vineyard.deploy.local


def test_local_instances():
    vineyard.deploy.local.try_init()
    client = vineyard.connect()
    obj_id = client.put(1024)
    client1 = vineyard.connect()
    assert client1.get(obj_id) == 1024
    client2 = vineyard.connect()
    assert client.instance_id == client2.instance_id
    vineyard.deploy.local.shutdown()


def test_local_instances_connect():
    client = vineyard.connect()
    obj_id = client.put(1024)
    client1 = vineyard.connect()
    assert client1.get(obj_id) == 1024
    client2 = vineyard.connect()
    assert client.instance_id == client2.instance_id
    vineyard.deploy.local.shutdown()
