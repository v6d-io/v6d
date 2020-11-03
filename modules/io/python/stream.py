#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
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
from urllib.parse import urlparse

import vineyard.io
from vineyard import ObjectID
from vineyard.launcher.script import ScriptLauncher

from vineyard.io.dataframe import DataframeStreamBuilder
import pyorc
import pyarrow as pa

base_path = os.path.abspath(os.path.dirname(__file__))


def _resolve_ssh_script(ssh=True):
    if ssh:
        stream_bash = os.path.join(base_path, 'ssh.sh')
    else:
        stream_bash = os.path.join(base_path, 'kube_ssh.sh')
    return stream_bash


class StreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script.
    '''
    def __init__(self, vineyard_endpoint=None, ssh=True):
        ''' Launch a job to read as a vineyard stream.

            Parameters
            ----------
            vineyard_endpoint: str
                IPC or RPC endpoint to connect to vineyard. If not specified, vineyard
                will try to discovery vineyardd from the environment variable named
                :code:`VINEYARD_IPC_SOCKET`.
        '''
        self.vineyard_endpoint = vineyard_endpoint
        super(StreamLauncher, self).__init__(_resolve_ssh_script(ssh=ssh))

    def wait(self, timeout=None):
        return vineyard.ObjectID(super(StreamLauncher, self).wait(timeout=timeout))


class ParallelStreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script, in which `ssh` or `kubectl exec` will
        be used under the hood.
    '''
    def __init__(self, ssh=True):
        self.vineyard_endpoint = None
        super(ParallelStreamLauncher, self).__init__(_resolve_ssh_script(ssh=ssh))

        self._streams = []
        self._procs = []

    def run(self, *args, **kwargs):
        ''' Execute a job to read as a vineyard stream or write a vineyard stream to
            external data sink.

            Parameters
            ----------
            vineyard_endpoint: str
                The local IPC or RPC endpoint to connect to vineyard. If not specified,
                vineyard will try to discovery vineyardd from the environment variable
                named :code:`VINEYARD_IPC_SOCKET` and :code:`VINEYARD_RPC_ENDPOINT`.
        '''
        self.vineyard_endpoint = kwargs.pop('vineyard_endpoint', None)
        if ':' in self.vineyard_endpoint:
            self.vineyard_endpoint = tuple(self.vineyard_endpoint.split(':'))

        num_workers = kwargs.pop('num_workers', 1)
        hosts = kwargs.pop('hosts', ['localhost'])

        nh = len(hosts)
        slots = [num_workers // nh + int(i < num_workers % nh) for i in range(nh)]
        proc_idx = 0
        for host, nproc in zip(hosts, slots):
            for _iproc in range(nproc):
                launcher = StreamLauncher()
                if not args:
                    args = (num_workers, proc_idx)
                else:
                    args = args + (num_workers, proc_idx)
                launcher.run(host, *args, **kwargs)
                proc_idx += 1
                self._procs.append(launcher)

    def join(self):
        for proc in self._procs:
            proc.join()

    def dispose(self, desired=True):
        for proc in self._procs:
            proc.dispose()

    def wait(self, timeout=None):
        partial_ids = []
        for proc in self._procs:
            r = proc.wait(timeout=timeout)
            partial_ids.append(r)
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::ParallelStream'
        meta['size_'] = len(partial_ids)
        for idx, partition_id in enumerate(partial_ids):
            meta.add_member('stream_%d' % idx, partition_id)
        return vineyard.connect(self.vineyard_endpoint).create_metadata(meta)


def get_executable(name):
    return f'vineyard_{name}'


def read_local_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a byte stream from local files.
    '''
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_local_bytes'),
                 *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_kafka_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a bytes stream from a kafka topic.
    '''
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_kafka_bytes'),
                 *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def parse_bytes_to_dataframe(byte_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('parse_bytes_to_dataframe'),
                 *((vineyard_socket, str(byte_stream)) + args), **kwargs)
    return launcher.wait()

def arrow_type(field):
    if field.name == 'decimal':
        return pa.decimal128(field.precision)
    elif field.name == 'uniontype':
        return pa.union(field.cont_types)
    elif field.name == 'array':
        return pa.list_(field.type)
    elif field.name == 'map':
        return pa.map_(field.key, field.value)
    elif field.name == 'struct':
        return pa.struct(field.fields)
    else:
        types = {
            'boolean': pa.bool_(),
            'tinyint': pa.int8(),
            'smallint': pa.int16(),
            'int': pa.int32(),
            'bigint': pa.int64(),
            'float': pa.float32(),
            'double': pa.float64(),
            'string': pa.string(),
            'char': pa.string(),
            'varchar': pa.string(),
            'binary': pa.binary(),
            'timestamp': pa.timestamp('ms'),
            'date': pa.date32(),
        }
        if field.name not in types:
            raise ValueError('Cannot convert to arrow type: ' + field.name)
        return types[field.name]

def read_local_orc(path, vineyard_socket, *args, **kwargs):
    client = vineyard.connect(vineyard_socket)
    builder = DataframeStreamBuilder(client)
    stream = builder.seal(client)
    writer = stream.open_writer(client)

    with open(path, 'rb') as f:
        reader = pyorc.Reader(f)
        fields = reader.schema.fields
        schema = []
        for c in fields:
            schema.append((c, arrow_type(fields[c])))
        pa_struct = pa.struct(schema)
        while rows := reader.read(num=1024):
            rb = pa.RecordBatch.from_struct_array(pa.array(rows, type=pa_struct))
            sink = pa.BufferOutputStream()
            rb_writer = pa.ipc.new_stream(sink, rb.schema)
            rb_writer.write_batch(rb)
            rb_writer.close()
            buf = sink.getvalue()
            chunk = writer.next(buf.size)
            buf_writer = pa.FixedSizeBufferWriter(chunk)
            buf_writer.write(buf)
            buf_writer.close()

    writer.finish()
    return stream

def write_local_orc(stream, vineyard_socket, *args, **kwargs):
    pass

def read_local_dataframe(path, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        return read_local_orc(path, vineyard_socket, *args, **kwargs)

    return parse_bytes_to_dataframe(read_local_bytes(path, vineyard_socket, *args, **kwargs), **kwargs)


def read_kafka_dataframe(path, vineyard_socket, **kwargs):
    return parse_bytes_to_dataframe(read_kafka_bytes(path, vineyard_socket, *args, **kwargs), **kwargs)


vineyard.io.read.register('file', read_local_bytes)
vineyard.io.read.register('file', read_local_dataframe)
vineyard.io.read.register('kafka', read_kafka_bytes)
vineyard.io.read.register('kafka', read_kafka_dataframe)

def write_local_dataframe(dataframe_stream, path, vineyard_socket, *args, **kwargs):
    if path.endswith('.orc'):
        write_local_orc(dataframe_stream, path, vineyard_socket, *args, **kwargs)
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_local_dataframe'),
                 path,
                 *((vineyard_socket, str(dataframe_stream)) + args), **kwargs)
    launcher.wait()

def write_kafka_bytes(dataframe_stream, path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_kafka_bytes'),
                 path,
                 *((vineyard_socket, str(dataframe_stream)) + args), **kwargs)
    launcher.wait()

def write_kafka_dataframe(dataframe_stream, path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_kafka_dataframe'),
                 path,
                 *((vineyard_socket, str(dataframe_stream)) + args), **kwargs)
    launcher.wait()

vineyard.io.write.register('file', write_local_dataframe)
vineyard.io.write.register('kafka', write_kafka_bytes)
vineyard.io.write.register('kafka', write_kafka_dataframe)
