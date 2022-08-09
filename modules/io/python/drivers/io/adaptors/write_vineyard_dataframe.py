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

import sys

import vineyard
from vineyard.io.dataframe import DataframeStream
from vineyard.io.utils import report_success


def write_vineyard_dataframe(vineyard_socket, stream_id, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(
            f"Fetch stream error with proc_num={proc_num},proc_index={proc_index}"
        )
    instream: DataframeStream = streams[proc_index]
    stream_reader = instream.open_reader(client)

    batch_index = 0
    while True:
        try:
            batch = stream_reader.next()
        except Exception:  # pylint: disable=broad-except
            break
        df = batch.to_pandas()
        df_id = client.put(
            df, partition_index=[proc_index, 0], row_batch_index=batch_index
        )
        batch_index += 1
        client.persist(df_id)
        report_success(df_id)


def main():
    if len(sys.argv) < 5:
        print(
            "usage: ./write_vineyard_dataframe <ipc_socket> <stream_id> "
            "<proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    write_vineyard_dataframe(ipc_socket, stream_id, proc_num, proc_index)


if __name__ == "__main__":
    main()
