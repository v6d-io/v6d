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
import sys
import time
import pandas as pd
import vineyard

client = vineyard.connect('/var/run/vineyard.sock')

env_dist = os.environ
objid = env_dist['OBJECT_ID']

df = pd.DataFrame({'a': [11, 12, 13, 14], 'b': [15, 16, 17, 18]})
obj = client.put(df)
client.persist(obj)

meta1 = client.get_meta(obj)
meta2 = client.get_meta(vineyard.ObjectID(objid))

meta = vineyard.ObjectMeta()
meta['__elements_-size'] = 2
meta['typename'] = 'vineyard::GlobalDataFrame'
meta.set_global(True)
meta.add_member('__elements_-0', meta1)
meta.add_member('__elements_-1', meta2)
tup = client.create_metadata(meta)
client.persist(tup)

# flush the stdout buffer
f = open('/dev/null', 'w') # pylint: disable=unspecified-encoding,consider-using-with
sys.stdout = f
print(flush=True)

sys.stdout = sys.__stdout__
print(tup.id, flush=True)

# avoid CrashLoopBackOff
time.sleep(1200)
