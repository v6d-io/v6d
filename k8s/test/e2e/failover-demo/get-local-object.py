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
import time
import numpy as np
import vineyard


client = vineyard.connect('/var/run/vineyard.sock')
env_dist = os.environ

objid = env_dist['OBJECT_ID']

value = client.get(vineyard.ObjectID(objid))

sum = np.sum(value)
print(sum,flush=True)

# avoid CrashLoopBackOff
time.sleep(600)
