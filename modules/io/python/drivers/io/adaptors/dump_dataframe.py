#! /usr/bin/env python3
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

import sys

import vineyard
from vineyard.io.dataframe import DataframeStream


def dump_dataframe(vineyard_socket, stream_id):
    client = vineyard.connect(vineyard_socket)
    stream: DataframeStream = client.get(stream_id)
    stream_reader = stream.open_reader(client)

    print('metadata: %s', stream.params)

    table = stream_reader.read_table()
    print('table: %s rows, %s columns' % (table.num_rows, table.num_columns))
    print('schema: %s' % (table.schema,))


def main():
    if len(sys.argv) < 5:
        print("usage: ./dump_dataframe <ipc_socket> <stream_id>")
        sys.exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    dump_dataframe(ipc_socket, stream_id)


if __name__ == "__main__":
    main()
