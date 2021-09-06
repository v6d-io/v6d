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

import filecmp
import itertools
import pytest

import vineyard
import vineyard.io


@pytest.mark.skip_without_migration()
def test_migrate_stream(vineyard_ipc_sockets, vineyard_endpoint, test_dataset, test_dataset_tmp):
    vineyard_ipc_sockets = list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), 2))

    # read the file as a stream, note that the open api
    # always returns a global stream
    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_sockets[0],
        vineyard_endpoint=vineyard_endpoint,
        read_options={
            "header_row": False,
            "delimiter": " "
        },
    )

    # extract the local stream from the opened global stream
    client1 = vineyard.connect(vineyard_ipc_sockets[0])
    local_streams = client1.get(stream)

    # migrate the local stream to another vineyardd
    client2 = vineyard.connect(vineyard_ipc_sockets[1])
    new_stream = client2.migrate_stream(local_streams[0].id)

    # create a global stream from the migrated local stream to fit
    # the open api
    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::ParallelStream'
    meta.set_global(True)
    meta['size_'] = 1
    meta.add_member("stream_0", new_stream)
    ret_meta = client2.create_metadata(meta)
    client2.persist(ret_meta)

    # output the global stream
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        ret_meta.id,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_sockets[1],
        vineyard_endpoint=vineyard_endpoint,
    )

    # check the equility
    assert filecmp.cmp("%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp)
