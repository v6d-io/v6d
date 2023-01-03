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

import os
import vineyard

vineyard_client = vineyard.connect('/var/run/vineyard.sock')
env_dist = os.environ

stream_id = env_dist['STREAM_ID']
stream = vineyard_client.get(stream_id)
reader = stream.reader

index = 0
global_meta = vineyard.ObjectMeta()
global_meta['typename'] = 'vineyard::Sequence'

while True:
    try:
        chunk = reader.next()
        obj = vineyard_client.put(chunk)
        meta = vineyard_client.get_meta(obj)
        local_meta = vineyard.ObjectMeta()
        local_meta['typename'] = meta['typename']
        local_meta['size'] = 1
        local_meta.add_member('__elements_-0', obj)
        local_meta['__elements_-size'] = 1
        local_meta.set_global(False)
        lid = vineyard_client.create_metadata(local_meta)
        vineyard_client.persist(lid)
        global_meta.add_member('__elements_-{}'.format(index), lid)
    except StopIteration:
        break
    index += 1

global_meta['size'] = index
global_meta['__elements_-size'] = index
global_meta.set_global(True)
gid = vineyard_client.create_metadata(global_meta)
vineyard_client.persist(gid)
