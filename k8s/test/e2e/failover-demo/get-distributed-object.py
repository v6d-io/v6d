# pylint: disable=django-not-configured
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
    second_meta = vineyard_client.get_meta(meta['__elements_-{}'.format(i)].id)  # pylint: disable=no-member
    if second_meta['typename'] != "vineyard::DataFrame":
        verified = False
        break
    for j in range(0,2):
        if second_meta['__values_-value-{}'.format(j)].nbytes != 32:
            verified = False
if verified:
    print('Passed', flush=True)
