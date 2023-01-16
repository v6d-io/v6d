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

import time
import os
import pandas as pd
import vineyard
from vineyard.io.recordbatch import RecordBatchStream


env_dist = os.environ
metaid = env_dist['METADATA_ID']

def generate_df(index):
    return pd.DataFrame({'a': [int(index), int(index)], 'b': [int(index), int(index)]})


vineyard_client = vineyard.connect('/var/run/vineyard.sock')

stream = RecordBatchStream.new(vineyard_client)
vineyard_client.persist(stream.id)
meta1 = vineyard_client.get_meta(stream.id)
meta2 = vineyard_client.get_meta(vineyard.ObjectID(metaid))

meta = vineyard.ObjectMeta()
meta['__elements_-size'] = 2
meta['typename'] = 'vineyard::GlobalDataFrameStream'
meta.set_global(True)
meta.add_member('__elements_-0', meta1)
meta.add_member('__elements_-1', meta2)
tup = vineyard_client.create_metadata(meta)
vineyard_client.persist(tup)

writer = stream.writer
total_chunks = 10
for idx in range(total_chunks):
    time.sleep(idx)
    chunk = generate_df(idx)
    chunk_id = vineyard_client.put(chunk)
    writer.append(chunk_id)

writer.finish()

print("writer finished",flush=True)

# avoid CrashLoopBackOff
time.sleep(3600)
