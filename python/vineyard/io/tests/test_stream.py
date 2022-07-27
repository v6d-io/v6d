#! /usr/bin/env python
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

import logging
import threading
import time
from typing import List

import numpy as np
import pandas as pd

from vineyard.core import default_builder_context
from vineyard.core import default_resolver_context
from vineyard.data import register_builtin_types
from vineyard.io.byte import ByteStream  # pylint: disable=unused-import
from vineyard.io.dataframe import DataframeStream  # pylint: disable=unused-import
from vineyard.io.recordbatch import RecordBatchStream

register_builtin_types(default_builder_context, default_resolver_context)

logger = logging.getLogger('vineyard')


def generate_random_dataframe(dtypes, size):
    columns = dict()
    for k, v in dtypes.items():
        columns[k] = np.random.random(size).astype(v)
    return pd.DataFrame(columns)


def test_recordbatch_stream(vineyard_client):
    total_chunks = 10

    def producer(stream: RecordBatchStream, dtypes, produced: List):
        writer = stream.writer
        for idx in range(total_chunks):
            time.sleep(idx)
            chunk = generate_random_dataframe(dtypes, 2)  # np.random.randint(10, 100))
            chunk_id = vineyard_client.put(chunk)
            writer.append(chunk_id)
            produced.append((chunk_id, chunk))
        writer.finish()

    def consumer(stream: RecordBatchStream, produced: List):
        reader = stream.reader
        index = 0
        while True:
            try:
                chunk = reader.next()
                pd.testing.assert_frame_equal(produced[index][1], chunk)
            except StopIteration:
                break
            index += 1

    stream = RecordBatchStream.new(vineyard_client)
    dtypes = {
        'a': np.dtype('int'),
        'b': np.dtype('float'),
        'c': np.dtype('bool'),
    }

    client1 = vineyard_client.fork()
    client2 = vineyard_client.fork()
    stream1 = client1.get(stream.id)
    stream2 = client2.get(stream.id)

    produced = []

    thread1 = threading.Thread(target=consumer, args=(stream1, produced))
    thread1.start()

    thread2 = threading.Thread(target=producer, args=(stream2, dtypes, produced))
    thread2.start()

    thread1.join()
    thread2.join()
