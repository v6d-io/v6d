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

import base64
import json
import logging
import sys

import pyarrow as pa
import pyarrow.csv  # pylint: disable=unused-import

import cloudpickle

import vineyard
from vineyard.data.utils import normalize_arrow_dtype
from vineyard.data.utils import str_to_bool
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
    vineyard_socket,
    stream_id,
    proc_num,
    proc_index,
    read_options: dict,
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

    # process parsing and converting options

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

    csv_read_options = pa.csv.ReadOptions()
    csv_parse_options = pa.csv.ParseOptions()
    csv_convert_options = pa.csv.ConvertOptions()

    if original_columns:
        csv_read_options.column_names = original_columns
    else:
        csv_read_options.autogenerate_column_names = True
    csv_parse_options.delimiter = delimiter

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
        csv_convert_options.include_columns = columns
    if len(column_types) > len(columns):
        raise ValueError("Format of column type schema is incorrect: too many columns")

    arrow_column_types = dict()
    for i, column_type in enumerate(column_types):
        if column_type:
            arrow_column_types[columns[i]] = normalize_arrow_dtype(column_type)
    csv_convert_options.column_types = arrow_column_types

    chunks = []
    if read_options.get('accumulate', False):
        stream_writer = None
    else:
        stream = DataframeStream.new(client, params=instream.params)
        client.persist(stream.id)
        report_success(stream.id)
        stream_writer = stream.open_writer(client)

    chunk_hook = read_options.get('chunk_hook', None)

    try:
        while True:
            try:
                content = stream_reader.next()
            except (StopIteration, vineyard.StreamDrainedException):
                if stream_writer is not None:
                    stream_writer.finish()
                else:
                    report_success(
                        json.dumps([repr(vineyard.ObjectID(k)) for k in chunks])
                    )
                break

            # parse csv
            table = parse_dataframe_blocks(
                content, csv_read_options, csv_parse_options, csv_convert_options
            )
            if chunk_hook is not None:
                batches = table.to_batches()
                new_batches = []
                for batch in batches:
                    new_batches.append(chunk_hook(batch))
                table = pa.Table.from_batches(new_batches)
            # write recordbatches
            if stream_writer is not None:
                stream_writer.write_table(table)
            else:
                for batch in table.to_batches():
                    chunks.append(client.put(batch.to_pandas(), persist=True))
    except Exception:  # pylint: disable=broad-except
        report_exception()
        stream_writer.fail()
        sys.exit(-1)

    # drop the stream
    if hasattr(instream, 'drop'):
        instream.drop(client)


def main():
    if len(sys.argv) < 5:
        print(
            "usage: ./parse_bytes_to_dataframe.py <ipc_socket> <stream_id> "
            "<read_options> <proc_num> <proc_index>"
        )
        sys.exit(1)
    ipc_socket = sys.argv[1]
    stream_id = sys.argv[2]
    try:
        read_options = json.loads(
            base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
        )
        if not isinstance(read_options, dict):
            read_options = {'accumulate': str_to_bool(read_options)}
    except:  # noqa: E722, pylint: disable=bare-except
        read_options = {'accumulate': str_to_bool(sys.argv[3])}
    if 'chunk_hook' in read_options:
        read_options['chunk_hook'] = cloudpickle.loads(
            base64.b64decode(read_options['chunk_hook'].encode('ascii'))
        )
    proc_num = int(sys.argv[4])
    proc_index = int(sys.argv[5])
    parse_bytes(ipc_socket, stream_id, proc_num, proc_index, read_options=read_options)


if __name__ == "__main__":
    main()
