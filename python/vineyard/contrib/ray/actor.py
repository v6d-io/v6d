#!/usr/bin/env python3
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

import contextlib

import ray
from ray.remote_function import RemoteFunction
from ray.util.placement_group import placement_group
from ray.util.placement_group import remove_placement_group


@contextlib.contextmanager
def spread_to_all_nodes(f: RemoteFunction):
    nodes = ray.state.nodes()
    resources = [{'CPU': f._num_cpus} for _ in range(len(nodes))]
    pg = placement_group(resources, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    yield len(nodes), pg
    remove_placement_group(pg)


def spread_and_get(f: RemoteFunction, *args, **kwargs):
    with spread_to_all_nodes(f) as (nodes, pg):
        return ray.get(
            [
                f.options(
                    placement_group=pg, placement_group_bundle_index=index
                ).remote(*args, **kwargs)
                for index in range(nodes)
            ]
        )


def spread(f: RemoteFunction, *args, **kwargs):
    with spread_to_all_nodes(f) as (nodes, pg):
        return [
            f.options(placement_group=pg, placement_group_bundle_index=index).remote(
                *args, **kwargs
            )
            for index in range(nodes)
        ]
