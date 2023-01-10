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
import json
import vineyard

vineyard_client = vineyard.connect('/var/run/vineyard.sock')

env_dist = os.environ
# map from old object to new object
allstr = env_dist['OldObjectToNewObject']
dist = json.loads(allstr)

globalobject_id = env_dist['GLOBALOBJECT_ID']
new_meta = vineyard.ObjectMeta()
new_meta_size = 0
new_id_size = 0
new_meta['typename'] = 'vineyard::GlobalSequence'
new_meta.set_global(True)
old_meta = vineyard_client.get_meta(vineyard.ObjectID(
    globalobject_id))  # pylint: disable=no-member
for i in range(0, old_meta['__elements_-size']):
    member_meta = vineyard_client.get_meta(vineyard.ObjectID(
        old_meta['__elements_-{}'.format(i)].id))  # pylint: disable=no-member
    if member_meta['id'] in dist:
        id = vineyard.ObjectID(dist[member_meta['id']])
        local_meta = vineyard_client.get_meta(id)  # pylint: disable=no-member
        for j in range(0, local_meta['__elements_-size']):
            md = local_meta['__elements_-{}'.format(j)]
            new_meta.add_member('__elements_-{}'.format(new_meta_size),
                                md.id)  # pylint: disable=no-member
            new_meta_size += 1
        new_id_size += 1
        # we can't use the add_member function here, because the member must be a local chunk
        new_meta['__global_id_-{}'.format(i)] = dist[member_meta['id']]
    else:
        new_meta.add_member('__global_id_-{}'.format(new_meta_size), member_meta.id)
        new_id_size += 1
        new_meta_size += 1

new_meta['__global_id_-size'] = new_id_size
new_meta['__elements_-size'] = new_meta_size
gid = vineyard_client.create_metadata(new_meta)
vineyard_client.persist(gid)
