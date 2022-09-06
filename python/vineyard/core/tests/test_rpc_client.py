#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

from ..._C import RemoteBlobBuilder
from ...core import default_builder_context
from ...core import default_resolver_context
from ...data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


payload = b'abcdefgh1234567890uvwxyz'


def test_remote_blob_create(vineyard_client, vineyard_endpoint):
    vineyard_rpc_client = vineyard.connect(*vineyard_endpoint.split(':'))

    buffer_writer = RemoteBlobBuilder(len(payload))
    buffer_writer.copy(0, payload)
    blob_id = vineyard_rpc_client.create_remote_blob(buffer_writer)

    # get as local blob
    local_blob = vineyard_client.get_blob(blob_id)

    # check local blob
    assert local_blob.id == blob_id
    assert local_blob.size == len(payload)
    assert memoryview(local_blob) == memoryview(payload)


def test_remote_blob_get(vineyard_client, vineyard_endpoint):
    vineyard_rpc_client = vineyard.connect(*vineyard_endpoint.split(':'))

    buffer_writer = vineyard_client.create_blob(len(payload))
    buffer_writer.copy(0, payload)
    blob = buffer_writer.seal(vineyard_client)

    # get as remote blob
    remote_blob = vineyard_rpc_client.get_remote_blob(blob.id)

    # check remote blob
    assert remote_blob.id == blob.id
    assert remote_blob.size == blob.size
    assert remote_blob.size == len(payload)
    assert memoryview(remote_blob) == memoryview(blob)
    assert memoryview(remote_blob) == memoryview(payload)


def test_remote_blob_create_and_get(vineyard_endpoint):
    vineyard_rpc_client = vineyard.connect(*vineyard_endpoint.split(':'))

    buffer_writer = RemoteBlobBuilder(len(payload))
    buffer_writer.copy(0, payload)
    blob_id = vineyard_rpc_client.create_remote_blob(buffer_writer)

    # get as remote blob
    remote_blob = vineyard_rpc_client.get_remote_blob(blob_id)

    # check remote blob
    assert remote_blob.id == blob_id
    assert remote_blob.size == len(payload)
    assert memoryview(remote_blob) == memoryview(payload)
