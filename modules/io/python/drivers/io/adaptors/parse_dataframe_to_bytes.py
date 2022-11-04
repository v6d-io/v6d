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

import base64
import json
import sys

import vineyard
from vineyard.io.byte import ByteStream
from vineyard.io.dataframe import DataframeStream
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def parse_dataframe(vineyard_socket, stream_id, write_options, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(
            f"Fetch stream error with proc_num={proc_num},proc_index={proc_index}"
        )
    instream: DataframeStream = streams[proc_index]
    stream_reader = instream.open_reader(client)

    header_row = None
    if "header_row" in write_options:
        header_row = write_options["header_row"]
    if header_row is None:
        if "header_row" in instream.params:
            header_row = instream.params["header_row"]
    if header_row is None:
        header_row = False
    else:
        header_row = strtobool(header_row)
    delimiter = None
    if 'delimiter' in write_options:
        delimiter = write_options['delimiter']
    if delimiter is None:
        if 'sep' in write_options:
            delimiter = write_options['sep']
    if delimiter is None:
        if 'delimiter' in instream.params:
            delimiter = instream.params['delimiter']
    if delimiter is None:
        delimiter = ','

    stream = ByteStream.new(client, params=instream.params)
    client.persist(stream.id)
    report_success(stream.id)

    stream_writer = stream.open_writer(client)
    first_write = header_row

    try:
        while True:
            try:
                batch = stream_reader.next()  # pa.RecordBatch
            except (StopIteration, vineyard.StreamDrainedException):
                stream_writer.finish()
                break
            df = batch.to_pandas()
            csv_content = df.to_csv(
                header=first_write, index=False, sep=delimiter
            ).encode('utf-8')

            # write to byte stream
            first_write = False
            if len(csv_content) > 0:
                chunk = stream_writer.next(len(csv_content))
                vineyard.memory_copy(chunk, 0, csv_content)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        stream_writer.fail()
        sys.exit(-1)


def main():
    if len(sys.argv) < 5:
        print(
            "usage: ./parse_dataframe_to_bytes <ipc_socket> <stream_id> "
            "<write_options> <proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    write_options = json.loads(
        base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
    )
    proc_num = int(sys.argv[4])
    proc_index = int(sys.argv[5])
    parse_dataframe(ipc_socket, stream_id, write_options, proc_num, proc_index)


if __name__ == "__main__":
    main()
