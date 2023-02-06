#! /usr/bin/env python3
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

import contextlib
import time
import os

import dask.array as da
import dask.dataframe as dd

import vineyard
from vineyard.contrib.dask.dask import register_dask_types
from vineyard.core.builder import builder_context
from vineyard.core.resolver import resolver_context


@contextlib.contextmanager
def vineyard_for_dask():
    with builder_context() as builder:
        with resolver_context() as resolver:
            register_dask_types(builder, resolver)
            yield builder, resolver


env_dist = os.environ
dask_scheduler = env_dist['DASK_SCHEDULER']
client = vineyard.connect('/var/run/vineyard.sock')

with vineyard_for_dask():
    arr = da.ones((1024, 2), chunks=(256, 2))
    df = dd.from_dask_array(arr, columns=['a', 'b'])
    obj_id = client.put(df, dask_scheduler=dask_scheduler)

# avoid CrashLoopBackOff
time.sleep(3600)
