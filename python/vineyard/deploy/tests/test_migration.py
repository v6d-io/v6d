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

import itertools
import json
import logging

import pytest
import numpy as np

import vineyard
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)

logger = logging.getLogger('vineyard')


@pytest.mark.skip_without_migration()
def test_migration(vineyard_ipc_sockets):
    vineyard_ipc_sockets = list(itertools.islice(itertools.cycle(vineyard_ipc_sockets), 2))

    client1 = vineyard.connect(vineyard_ipc_sockets[0])
    client2 = vineyard.connect(vineyard_ipc_sockets[1])

    # test it metadata of remote object available
    data = np.ones((1, 2, 3, 4, 5))
    o = client1.put(data)
    client1.persist(o)
    meta = client2.get_meta(o)
    assert data.shape == tuple(json.loads(meta['shape_']))

    # migrate local to local: do nothing.
    o1 = client1.migrate(o)
    assert o == o1
    logger.info('------- finish round 1 --------')

    # migrate remote to local: do nothing.
    o2 = client2.migrate(o)
    assert o != o2
    np.testing.assert_allclose(client1.get(o1), client2.get(o2))
    logger.info('------- finish round 2 --------')
