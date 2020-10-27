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
import pkg_resources
from urllib.parse import urlparse

import vineyard
from vineyard import ObjectID
from vineyard.launcher.script import ScriptLauncher


class StreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script.
    '''
    def __init__(self):
        stream_bash = pkg_resources.resource_filename('vineyard.io', 'scripts/stream_bash.sh')
        super(StreamLauncher, self).__init__(stream_bash)


class ParallelStreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script, in which `mpirun` will be used.
    '''
    def __init__(self, num_workers):
        stream_mpi = pkg_resources.resource_filename('vineyard.io', 'scripts/stream_mpi.sh')
        super(ParallelStreamLauncher, self).__init__(stream_mpi)

        self._num_workers = num_workers
        self._streams = []

    def run(self, *args, **kwargs):
        num_workers = kwargs.pop('num_workers', self._num_workers)
        hosts = kwargs.pop('hosts', ['localhost'])
        if num_workers >= len(hosts):
            nh = len(hosts)
            slots = [1 if i < num_workers % nh else 0 for i in range(nh)]
            slots = [num_workers // nh + i for i in slots]
            hosts = ['%s:%s' % (h, slot) for h, slot in zip(hosts, slots)]
        else:
            hosts = hosts[:num_workers]
        kwargs['WORKER_NUM'] = str(num_workers)
        kwargs['HOSTS'] = ','.join(hosts)
        return super(ParallelStreamLauncher, self).run(*args, **kwargs)

    @property
    def streams(self):
        return self._streams

    def wait_many(self):
        ''' Wait util we collect enough partial stream object ids.
        '''
        while len(self._streams) < self._num_workers:
            self._streams.append(super(ParallelStreamLauncher, self).wait())
        return self._streams


def get_executable(name):
    return pkg_resources.resource_filename('vineyard.io.binaries', f'vineyard_{name}.bin')


def single_run(client, path, executable, *args, **kwargs):
    launcher = StreamLauncher()
    launcher.run(get_executable(executable), client.ipc_socket, path, *args, **kwargs)
    r = launcher.wait()
    return client.get_object(ObjectID(r))


def single_dataframe_parser(client, byte_stream, **kwargs):
    launcher = StreamLauncher()
    launcher.run(get_executable('single_dataframe_parser'), client.ipc_socket, byte_stream.id, **kwargs)
    r = launcher.wait()
    return client.get_object(ObjectID(r))


def single_local_byte(client, path, **kwargs):
    ''' Read a bytes stream from local files.
    '''
    return single_run(client, path, "single_local_byte", **kwargs)


def single_kafka_byte(client, path, **kwargs):
    ''' Read a bytes stream from a kafka topic.
    '''
    return single_run(client, path, 'single_kafka_byte', **kwargs)


def single_oss_dataframe(client, path, **kwargs):
    ''' Read a bytes stream from OSS object storage.
    '''
    return single_run(client, path, 'single_oss_dataframe', **kwargs)


def single_local_dataframe(client, path, **kwargs):
    return single_dataframe_parser(client, single_local_byte(client, path, **kwargs), **kwargs)


def single_kafka_dataframe(client, path, **kwargs):
    return single_dataframe_parser(client, single_kafka_byte(client, path), **kwargs)


def parallel_local_byte(client, path, num_workers=4, **kwargs):
    launcher = ParallelStreamLauncher(num_workers)
    launcher.run(get_executable('parallel_local_byte'), client.ipc_socket, path, **kwargs)
    r = launcher.wait()
    return client.get_object(ObjectID(r))


def parallel_dataframe_parser(client, byte_stream, num_workers=4, **kwargs):
    launcher = ParallelStreamLauncher(num_workers)
    launcher.run(get_executable('parallel_dataframe_parser'), client.ipc_socket, byte_stream.id, **kwargs)
    r = launcher.wait()
    return client.get_object(ObjectID(r))


def parallel_local_dataframe(client, path, **kwargs):
    return parallel_dataframe_parser(client, parallel_local_byte(client, path, **kwargs), **kwargs)


def single_dataframe_dataframe(client, path, **kwargs):
    object_id = ObjectID(urlparse(path).netloc)
    launcher = StreamLauncher()
    launcher.run(get_executable('single_dataframe_dataframe'), client.ipc_socket, object_id, **kwargs)
    r = launcher.wait()
    return client.get_object(ObjectID(r))


vineyard.read.register('single', 'vineyard', single_dataframe_dataframe)
vineyard.read.register('parallel', 'file', parallel_local_dataframe)
vineyard.read.register('single', 'file', single_local_byte)
vineyard.read.register('single', 'file', single_local_dataframe)
vineyard.read.register('single', 'oss', single_oss_dataframe)
vineyard.read.register('single', 'kafka', single_kafka_byte)


def write_file(client, executable, ofile, stream):
    launcher = StreamLauncher()
    launcher.run(get_executable(executable), client.ipc_socket, stream.id, ofile)
    launcher.join()


def parallel_write_file(client, ofile, stream):
    write_file(client, 'parallel_dataframe_single_local_consumer', ofile, stream)


def single_write_file(client, ofile, stream):
    write_file(client, 'single_dataframe_single_local_consumer', ofile, stream)


def single_write_kafka(client, path, stream):
    write_file(client, 'single_byte_single_kafka_consumer', path, stream)


vineyard.write.register('parallel', 'file', parallel_write_file)
vineyard.write.register('single', 'file', single_write_file)
vineyard.write.register('single', 'kafka', single_write_kafka)
