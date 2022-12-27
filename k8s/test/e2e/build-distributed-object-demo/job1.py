#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

import time
import numpy as np
import vineyard

client = vineyard.connect('/var/run/vineyard.sock')

obj = client.put(np.arange(1,6))
client.persist(obj)
# flush the stdout buffer
f = open('/dev/null', 'w') # pylint: disable=unspecified-encoding,consider-using-with
sys.stdout = f
print(flush=True)

sys.stdout = sys.__stdout__
print(obj.id, flush=True)

# avoid CrashLoopBackOff
time.sleep(600)
