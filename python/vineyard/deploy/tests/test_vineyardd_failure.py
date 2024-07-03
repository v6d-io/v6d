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

import numpy as np

import vineyard


def test_put_meta_before_failure(vineyard_endpoint):
    client = vineyard.connect(endpoint=vineyard_endpoint)

    for i in range(10):
        data = np.ones((1, 2, 3, 4, 5))
        client.put(data, name="data-%d" % i, persist=True)


def test_get_meta_after_failure(vineyard_endpoint):
    client = vineyard.connect(endpoint=vineyard_endpoint)

    for i in range(10):
        obj = client.get_name("data-%d" % i)
        client.get_meta(obj)
