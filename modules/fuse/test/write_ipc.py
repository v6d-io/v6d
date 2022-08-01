#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 Alibaba Group Holding Limited.
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
import vineyard as vy
import pyarrow as pa
import numpy as np
import pandas as pd
import sys
# default_socket = "/var/run/vineyard.sock"

# client = vy.connect(default_socket);
filename = sys.argv[1]
height,width = 5,4
df = pd.DataFrame(np.random.randint(0,100,size=(height, width)), columns=list('ABCD'))
pdf = pa.Table.from_pandas(df)

with open(filename, 'wb') as source:
    with pa.ipc.RecordBatchStreamWriter(source,pdf.schema) as writer:
        data = writer.write(pdf)
