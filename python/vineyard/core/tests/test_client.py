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


def test_metadata(vineyard_client):
    xid = vineyard_client.put(1.2345)
    yid = vineyard_client.put(2.3456)
    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', yid)
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)

    def go(meta):
        for k, v in meta.items():
            if isinstance(v, vineyard.ObjectMeta):
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
    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', yid)
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)


def test_persist_multiref(vineyard_client):
    xid = vineyard_client.put(1.2345)
    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::Pair'
    meta.add_member('first_', xid)
    meta.add_member('second_', xid)
    meta.set_global(True)
    rmeta = vineyard_client.create_metadata(meta)
    vineyard_client.persist(rmeta)
