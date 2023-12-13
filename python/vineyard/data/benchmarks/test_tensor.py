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
import pyarrow as pa

try:
    import pyarrow.plasma as plasma
except ImportError:
    plasma = None

import pytest
import pytest_cases

from vineyard.conftest import vineyard_client
from vineyard.conftest import vineyard_rpc_client
from vineyard.core import default_builder_context
from vineyard.core import default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


@pytest.fixture(scope="module", autouse=True)
def plasma_client():
    if plasma is None:
        pytest.skip("plasma is not installed, pyarrow<=11 is required")

    with plasma.start_plasma_store(plasma_store_memory=1024 * 1024 * 1024 * 8) as (
        plasma_socket,
        plasma_proc,
    ):
        plasma_client = plasma.connect(plasma_socket)
        yield plasma_client
        plasma_client.disconnect()
        plasma_proc.kill()


@pytest.mark.skipif(
    plasma is None, reason="plasma is not installed, pyarrow<=11 is required"
)
@pytest_cases.parametrize(
    "client,nbytes",
    [
        (vineyard_client, '256'),
        (vineyard_client, '256KB'),
        (vineyard_client, '256MB'),
        (vineyard_client, '1GB'),
        (vineyard_client, '4GB'),
        (plasma_client, '256'),
        (plasma_client, '256KB'),
        (plasma_client, '256MB'),
        (plasma_client, '1GB'),
        (plasma_client, '4GB'),
    ],
)
def test_bench_numpy_ndarray(benchmark, client, nbytes):
    shape = {
        '256': (64,),
        '256KB': (64, 1024),
        '256MB': (64, 1024, 1024),
        '1GB': (256, 1024, 1024),
        '4GB': (1024, 1024, 1024),
    }[nbytes]
    data = np.random.rand(*shape).astype(np.float32)

    def bench_numpy_ndarray(client, data):
        object_id = client.put(data)
        client.delete([object_id])

    benchmark(bench_numpy_ndarray, client, data)
