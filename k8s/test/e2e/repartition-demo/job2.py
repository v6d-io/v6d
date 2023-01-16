#! /usr/bin/env python3 # pylint: disable=missing-module-docstring
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

import os
import json
import contextlib
import time

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


client = vineyard.connect('/var/run/vineyard.sock')
env_dist = os.environ

job = env_dist['REQUIRED_JOB_NAME']
gid = env_dist.get(job)
# map from vineyard instances to dask workers
allstr = env_dist['InstanceToWorker']

dist = json.loads(allstr)

dask_workers = {}
for key, value in dist.items():
    dask_workers[int(key)] = value

dask_scheduler = env_dist['DASK_SCHEDULER']
with vineyard_for_dask():
    ddf = client.get(vineyard.ObjectID(gid),
                     dask_scheduler=dask_scheduler, dask_workers=dask_workers)
    partitions = ddf.npartitions
    print(partitions, flush=True)

# avoid CrashLoopBackOff
time.sleep(3600)
