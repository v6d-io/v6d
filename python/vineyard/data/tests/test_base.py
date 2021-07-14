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

import pytest

import vineyard
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


def test_int(vineyard_client):
    object_id = vineyard_client.put(1)
    assert vineyard_client.get(object_id) == 1


def test_double(vineyard_client):
    object_id = vineyard_client.put(1.234)
    assert vineyard_client.get(object_id) == pytest.approx(1.234)


def test_string(vineyard_client):
    object_id = vineyard_client.put('abcde')
    assert vineyard_client.get(object_id) == 'abcde'


def test_bytes(vineyard_client):
    bs = b'abcde'
    object_id = vineyard_client.put(bs)
    assert vineyard_client.get(object_id) == memoryview(bs)


def test_memoryview(vineyard_client):
    bs = memoryview(b'abcde')
    object_id = vineyard_client.put(bs)
    assert vineyard_client.get(object_id) == bs


def test_pair(vineyard_client):
    object_id = vineyard_client.put((1, "2"))
    assert vineyard_client.get(object_id) == (1, "2")


def test_tuple(vineyard_client):
    object_id = vineyard_client.put(())
    assert vineyard_client.get(object_id) == ()

    object_id = vineyard_client.put((1, ))
    assert vineyard_client.get(object_id) == (1, )

    object_id = vineyard_client.put((1, "2"))
    assert vineyard_client.get(object_id) == (1, "2")

    object_id = vineyard_client.put((1, "2", 3.456))
    assert vineyard_client.get(object_id) == (1, "2", pytest.approx(3.456))

    object_id = vineyard_client.put((1, "2", 3.456, 4444))
    assert vineyard_client.get(object_id) == (1, "2", pytest.approx(3.456), 4444)

    object_id = vineyard_client.put((1, "2", 3.456, 4444, "5.5.5.5.5.5.5"))
    assert vineyard_client.get(object_id) == (1, "2", pytest.approx(3.456), 4444, "5.5.5.5.5.5.5")
