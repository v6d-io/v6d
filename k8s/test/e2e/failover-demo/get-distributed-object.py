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

objid = env_dist['OBJECT_ID']

meta = vineyard_client.get_meta(vineyard._C.ObjectID(objid))  # pylint: disable=no-member
verified = True
for i in range(0, meta['__elements_-size']):
    oid = meta['__elements_-{}'.format(i)].id
    value = vineyard_client.get(vineyard_client.migrate(oid))
    sum += (value['a'].sum() + value['b'].sum())
print(sum, flush=True)
