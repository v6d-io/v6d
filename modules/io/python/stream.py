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

import json
import logging
import os
from urllib.parse import urlparse

import vineyard.io
from vineyard._C import ObjectID
from vineyard.launcher.script import ScriptLauncher

logger = logging.getLogger('vineyard')
base_path = os.path.abspath(os.path.dirname(__file__))

from vineyard.core.resolver import default_resolver_context


def parallel_stream_resolver(obj):
    ''' Return a list of *local* partial streams.
    '''
    meta = obj.meta
    partition_size = int(meta['size_'])
    streams = [meta.get_member('stream_%d' % i) for i in range(partition_size)]
    return [stream for stream in streams if stream.islocal]


default_resolver_context.register('vineyard::ParallelStream', parallel_stream_resolver)


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
        logger.debug('partial_ids = %s', partial_ids)
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::ParallelStream'
        meta['size_'] = len(partial_ids)
        for idx, partition_id in enumerate(partial_ids):
            meta.add_member('stream_%d' % idx, partition_id)
        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        return vineyard_rpc_client.create_metadata(meta)


def get_executable(name):
    return f'vineyard_{name}'


def read_local_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a byte stream from local files.
    '''
    path = json.dumps(path)
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_local_bytes'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_kafka_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a bytes stream from a kafka topic.
    '''
    path = json.dumps(path)
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_kafka_bytes'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def parse_bytes_to_dataframe(vineyard_socket, byte_stream, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('parse_bytes_to_dataframe'), *((vineyard_socket, byte_stream) + args), **kwargs)
    r = launcher.wait()
    logger.debug('parse r = %s', r)
    return r


def read_local_orc(path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_local_orc'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_local_dataframe(path, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        return read_local_orc(path, vineyard_socket, *args, **kwargs)
    return parse_bytes_to_dataframe(vineyard_socket, read_local_bytes(path, vineyard_socket, *args, **kwargs), *args, **kwargs)


def read_kafka_dataframe(path, vineyard_socket, *args, **kwargs):
    return parse_bytes_to_dataframe(vineyard_socket, read_kafka_bytes(path, vineyard_socket, *args, **kwargs), *args, **kwargs)


def read_hdfs_bytes(path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_hdfs_bytes'), *((vineyard_socket, 'hdfs://' + path) + args), **kwargs)
    return launcher.wait()

def read_hdfs_orc(path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('read_hdfs_orc'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_hdfs_dataframe(path, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        return read_hdfs_orc(path, vineyard_socket, *args, **kwargs)
    return parse_bytes_to_dataframe(vineyard_socket, read_hdfs_bytes(path, vineyard_socket, *args, **kwargs), *args, **kwargs)

def read_hive_dataframe(path, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    # Note that vineyard currently supports hive tables stored as orc format only
    launcher.run(get_executable('read_hive_orc'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()

vineyard.io.read.register('file', read_local_bytes)
vineyard.io.read.register('file', read_local_dataframe)
vineyard.io.read.register('kafka', read_kafka_bytes)
vineyard.io.read.register('kafka', read_kafka_dataframe)
vineyard.io.read.register('hdfs', read_hdfs_dataframe)
vineyard.io.read.register('hive', read_hive_dataframe)


def write_local_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_local_orc'), *((vineyard_socket, path, dataframe_stream) + args), **kwargs)
    launcher.join()


def write_local_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        write_local_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs)
        return
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_local_dataframe'), *((vineyard_socket, path, dataframe_stream) + args),
                 **kwargs)
    launcher.join()


def write_kafka_bytes(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_kafka_bytes'), *((vineyard_socket, path, dataframe_stream) + args),
                 **kwargs)
    launcher.join()


def write_kafka_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_kafka_dataframe'), *((vineyard_socket, path, dataframe_stream) + args),
                 **kwargs)
    launcher.join()


def write_hdfs_bytes(path, byte_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_hdfs_bytes'), *((vineyard_socket, 'hdfs://' + path, byte_stream) + args), **kwargs)
    launcher.join()

def write_hdfs_orc(path, byte_stream, vineyard_socket, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('write_hdfs_orc'), *((vineyard_socket, 'hdfs://' + path, byte_stream) + args), **kwargs)
    launcher.join()


def parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs):
    launcher = ParallelStreamLauncher()
    launcher.run(get_executable('parse_dataframe_to_bytes'), *((vineyard_socket, dataframe_stream) + args), **kwargs)
    return launcher.wait()


def write_hdfs_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        write_hdfs_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs)
    write_hdfs_bytes(path, parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs), vineyard_socket, *args, **kwargs)


vineyard.io.write.register('file', write_local_dataframe)
vineyard.io.write.register('kafka', write_kafka_bytes)
vineyard.io.write.register('kafka', write_kafka_dataframe)
vineyard.io.write.register('hdfs', write_hdfs_dataframe)
