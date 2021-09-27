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

try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.types as types
except ImportError:
    dali = None

from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context
from vineyard.contrib.ml.dali import register_dali_types


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_dali():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_dali_types(builder, resolver)
            yield builder, resolver


num_gpus = 1
device_id = 0
batch_size = 2
num_threads = 4


@pytest.mark.skip(dali is None, reason="nvidia-dali is not available")
def test_dali_tensor(vineyard_client):
    @pipeline_def()
    def pipe():
        data = np.array([np.random.rand(1, 2) for i in range(10)])
        label = np.array([np.random.rand(1, 3) for i in range(10)])
        fdata = types.Constant(data)
        flabel = types.Constant(label)
        return fdata, flabel

    pipeline = pipe(device_id=device_id, num_threads=num_threads, batch_size=batch_size)
    pipeline.build()
    pipe_out = pipeline.run()
    object_id = vineyard_client.put(pipe_out)
    pipe_vin = vineyard_client.get(object_id)
    assert pipe_vin[0][0].as_array().shape == pipe_out[0].as_array().shape
    assert pipe_vin[1][0].as_array().shape == pipe_out[1].as_array().shape
