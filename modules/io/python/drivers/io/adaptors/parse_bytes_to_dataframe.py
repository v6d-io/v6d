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

import logging
import sys

import pyarrow as pa
import pyarrow.csv  # pylint: disable=unused-import

import vineyard
from vineyard.data.utils import normalize_arrow_dtype
from vineyard.io.byte import ByteStream
from vineyard.io.dataframe import DataframeStream
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

logger = logging.getLogger('vineyard')


def parse_dataframe_blocks(content, read_options, parse_options, convert_options):
    reader = pa.BufferReader(content)
    return pa.csv.read_csv(reader, read_options, parse_options, convert_options)


def parse_bytes(  # noqa: C901, pylint: disable=too-many-statements
    vineyard_socket, stream_id, proc_num, proc_index
):
    client = vineyard.connect(vineyard_socket)

    # get input streams
    streams = client.get(stream_id)
    if len(streams) != proc_num or streams[proc_index] is None:
        raise ValueError(
            f"Fetch stream error with proc_num={proc_num},proc_index={proc_index}"
        )
    instream: ByteStream = streams[proc_index]
    stream_reader = instream.open_reader(client)

    use_header_row = instream.params.get("header_row", None) == "1"
    delimiter = instream.params.get("delimiter", ",")

    # process parsing and coverting options

    columns = []
    column_types = []
    original_columns = []
    header_line = None

    if use_header_row:
        header_line: str = instream.params.get('header_line', None)
        if not header_line:
            report_error('Header line not found while header_row is set to True')
            sys.exit(-1)
        original_columns = header_line.strip().split(delimiter)

    schema = instream.params.get('schema', None)
    if schema:
        columns = schema.split(',')

    column_types = instream.params.get('column_types', [])
    if column_types:
        column_types = column_types.split(',')

    include_all_columns = instream.params.get('include_all_columns', None) == '1'

    read_options = pa.csv.ReadOptions()
    parse_options = pa.csv.ParseOptions()
    convert_options = pa.csv.ConvertOptions()

    if original_columns:
        read_options.column_names = original_columns
    else:
        read_options.autogenerate_column_names = True
    parse_options.delimiter = delimiter

    indices = []
    for i, column in enumerate(columns):
        if original_columns:
            if column.isdigit():
                column_index = int(column)
                if column_index >= len(original_columns):
                    raise IndexError(
                        'Column index out of range: %s of %s'
                        % (column_index, original_columns)
                    )
                indices.append(i)
                columns[i] = original_columns[column_index]
        else:
            columns[i] = 'f%s' % i  # arrow auto generates column names in that way.

    if include_all_columns:
        for column in original_columns:
            if column not in columns:
                columns.append(column)

    if columns:
        convert_options.include_columns = columns
    if len(column_types) > len(columns):
        raise ValueError("Format of column type schema is incorrect: too many columns")

    arrow_column_types = dict()
    for i, column_type in enumerate(column_types):
        if column_type:
            arrow_column_types[columns[i]] = normalize_arrow_dtype(column_type)
    convert_options.column_types = arrow_column_types

    stream = DataframeStream.new(client, params=instream.params)
    client.persist(stream.id)
    report_success(stream.id)

    stream_writer = stream.open_writer(client)

    try:
        while True:
            try:
                content = stream_reader.next()
            except (StopIteration, vineyard.StreamDrainedException):
                stream_writer.finish()
                break

            # parse csv
            table = parse_dataframe_blocks(
                content, read_options, parse_options, convert_options
            )
            # write recordbatches
            stream_writer.write_table(table)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        stream_writer.fail()
        sys.exit(-1)


def main():
    if len(sys.argv) < 5:
        print(
            "usage: ./parse_bytes_to_dataframe.py <ipc_socket> <stream_id> "
            "<proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    parse_bytes(ipc_socket, stream_id, proc_num, proc_index)


if __name__ == "__main__":
    main()
