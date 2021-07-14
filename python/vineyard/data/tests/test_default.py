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

import numpy as np
import pytest

import vineyard
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


def test_bool(vineyard_client):
    value = True
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value

    value = False
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value


def test_np_bool(vineyard_client):
    value = np.bool_(True)
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value

    value = np.bool_(False)
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value


def test_list(vineyard_client):
    value = [1, 2, 3, 4, 5, 6, None, None, 9]
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value


def test_dict(vineyard_client):
    value = {1: 2, 3: 4, 5: None, None: 6}
    object_id = vineyard_client.put(value)
    assert vineyard_client.get(object_id) == value
