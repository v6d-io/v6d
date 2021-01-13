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

import vineyard.io
from vineyard.launcher.script import ScriptLauncher

logger = logging.getLogger('vineyard')
base_path = os.path.abspath(os.path.dirname(__file__))

from vineyard.core.resolver import default_resolver_context


def parallel_stream_resolver(obj):
    ''' Return a list of *local* partial streams.
    '''
    meta = obj.meta
    partition_size = int(meta['size_'])
    return [meta.get_member('stream_%d' % i) for i in range(partition_size)]


def global_dataframe_resolver(obj, resolver):
    ''' Return a list of dataframes.
    '''
    meta = obj.meta
    object_set = meta.get_member('objects_')
    meta = object_set.meta
    num = int(meta['num_of_objects'])
    dataframes = []
    orders = []
    for i in range(num):
        df = meta.get_member('object_%d' % i)
        if df.meta.islocal:
            dataframes.append(resolver.run(df))
            orders.append(df.meta['row_batch_index_'])
    if orders != sorted(orders):
        raise ValueError('Bad dataframe orders:', orders)
    return dataframes


default_resolver_context.register('vineyard::ParallelStream', parallel_stream_resolver)
default_resolver_context.register('vineyard::GlobalDataFrame', global_dataframe_resolver)


def _resolve_ssh_script(deployment='ssh'):
    if deployment == 'ssh':
        return os.path.join(base_path, 'ssh.sh')
    if deployment == 'kubernetes':
        return os.path.join(base_path, 'kube_ssh.sh')
    raise ValueError('Unknown deployment: "%s"' % deployment)


class StreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script.
    '''
    def __init__(self, vineyard_endpoint=None, deployment='ssh'):
        ''' Launch a job to read as a vineyard stream.

            Parameters
            ----------
            vineyard_endpoint: str
                IPC or RPC endpoint to connect to vineyard. If not specified, vineyard
                will try to discovery vineyardd from the environment variable named
                :code:`VINEYARD_IPC_SOCKET`.
        '''
        self.vineyard_endpoint = vineyard_endpoint
        super(StreamLauncher, self).__init__(_resolve_ssh_script(deployment=deployment))

    def wait(self, timeout=None):
        return vineyard.ObjectID(super(StreamLauncher, self).wait(timeout=timeout))


class ParallelStreamLauncher(ScriptLauncher):
    ''' Launch the job by executing a script, in which `ssh` or `kubectl exec` will
        be used under the hood.
    '''
    def __init__(self, deployment='ssh'):
        self.deployment = deployment
        self.vineyard_endpoint = None
        super(ParallelStreamLauncher, self).__init__(_resolve_ssh_script(deployment=deployment))

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
        kwargs = kwargs.copy()
        self.vineyard_endpoint = kwargs.pop('vineyard_endpoint', None)
        if ':' in self.vineyard_endpoint:
            self.vineyard_endpoint = tuple(self.vineyard_endpoint.split(':'))

        hosts = kwargs.pop('hosts', ['localhost'])
        num_workers = kwargs.pop('num_workers', len(hosts))

        nh = len(hosts)
        slots = [num_workers // nh + int(i < num_workers % nh) for i in range(nh)]
        proc_idx = 0
        for host, nproc in zip(hosts, slots):
            for _iproc in range(nproc):
                launcher = StreamLauncher(deployment=self.deployment)
                if not args:
                    proc_args = (num_workers, proc_idx)
                else:
                    proc_args = args + (num_workers, proc_idx)
                launcher.run(host, *proc_args, **kwargs)
                proc_idx += 1
                self._procs.append(launcher)

    def join(self):
        for proc in self._procs:
            proc.join()

    def dispose(self, desired=True):
        for proc in self._procs:
            proc.dispose()

    def wait(self, timeout=None, func=None):
        partial_ids = []
        for proc in self._procs:
            r = proc.wait(timeout=timeout)
            partial_ids.append(r)
        logger.debug('partial_ids = %s', partial_ids)
        if func is None:
            return self.create_parallel_stream(partial_ids)
        return func(partial_ids)

    def create_parallel_stream(self, partial_ids):
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::ParallelStream'
        meta.set_global(True)
        meta['size_'] = len(partial_ids)
        for idx, partition_id in enumerate(partial_ids):
            meta.add_member('stream_%d' % idx, partition_id)
        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        ret_id = vineyard_rpc_client.create_metadata(meta)
        vineyard_rpc_client.persist(ret_id)
        return ret_id

    def wait_all(self, func=None, **kwargs):
        partial_id_matrix = []
        for proc in self._procs:
            proc.join()
            partial_id_matrix.append([vineyard.ObjectID(ret_id) for ret_id in proc._result])
        logger.debug('partial_id_matrix = %s', partial_id_matrix)
        if func is None:
            return self.create_global_dataframe(partial_id_matrix, **kwargs)
        return func(partial_id_matrix, **kwargs)

    def create_global_dataframe(self, partial_id_matrix, **kwargs):
        # use the partial_id_matrix and the name in **kwargs
        # to create a global dataframe. Here the name is given in the
        # the input URI path in the
        # form of vineyard://{name_for_the_global_dataframe}
        name = kwargs.pop('name', None)
        if name is None:
            raise ValueError('Name of the global dataframe is not provided')
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::GlobalDataFrame'
        meta.set_global(True)
        meta['partition_shape_row_'] = len(partial_id_matrix)
        meta['partition_shape_column_'] = 1

        partition_size = 0
        for partial_id_list in partial_id_matrix:
            for partial_id in partial_id_list:
                meta.add_member('partitions__%d' % partition_size, partial_id)
                partition_size += 1
        meta['partitions_-size'] = partition_size

        meta['nbytes'] = 0  # FIXME
        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        gdf = vineyard_rpc_client.create_metadata(meta)
        vineyard_rpc_client.persist(gdf)
        vineyard_rpc_client.put_name(gdf, name)


def get_executable(name):
    return f'vineyard_{name}'


def parse_bytes_to_dataframe(vineyard_socket, byte_stream, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('parse_bytes_to_dataframe'), *((vineyard_socket, byte_stream) + args), **kwargs)
    return launcher.wait()


def read_local_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a byte stream from local files.
    '''
    path = json.dumps(path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_local_bytes'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_kafka_bytes(path, vineyard_socket, *args, **kwargs):
    ''' Read a bytes stream from a kafka topic.
    '''
    path = json.dumps(path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_kafka_bytes'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_local_orc(path, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_local_orc'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_local_dataframe(path, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        return read_local_orc(path, vineyard_socket, *args, **kwargs)
    return parse_bytes_to_dataframe(vineyard_socket, read_local_bytes(path, vineyard_socket, *args, **kwargs.copy()),
                                    *args, **kwargs.copy())


def read_kafka_dataframe(path, vineyard_socket, *args, **kwargs):
    return parse_bytes_to_dataframe(vineyard_socket, read_kafka_bytes(path, vineyard_socket, *args, **kwargs.copy()),
                                    *args, **kwargs.copy())


def read_hdfs_bytes(path, vineyard_socket, *args, **kwargs):
    path = json.dumps('hdfs://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_hdfs_bytes'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_hdfs_orc(path, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_hdfs_orc'), *((vineyard_socket, 'hdfs://' + path) + args), **kwargs)
    return launcher.wait()


def read_hdfs_dataframe(path, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        return read_hdfs_orc(path, vineyard_socket, *args, **kwargs)
    return parse_bytes_to_dataframe(vineyard_socket, read_hdfs_bytes(path, vineyard_socket, *args, **kwargs.copy()),
                                    *args, **kwargs.copy())


def read_hive_dataframe(path, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    # Note that vineyard currently supports hive tables stored as orc format only
    launcher.run(get_executable('read_hive_orc'), *((vineyard_socket, 'hive://' + path) + args), **kwargs)
    return launcher.wait()


def read_vineyard_dataframe(path, vineyard_socket, *args, **kwargs):
    path = json.dumps('vineyard://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    # Note that vineyard currently supports hive tables stored as orc format only
    launcher.run(get_executable('read_vineyard_dataframe'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


def read_oss_dataframe(path, vineyard_socket, *args, **kwargs):
    ''' Read a dataframe stream from oss files.
    '''
    path = json.dumps('oss://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('read_oss_dataframe'), *((vineyard_socket, path) + args), **kwargs)
    return launcher.wait()


vineyard.io.read.register('file', read_local_bytes)
vineyard.io.read.register('file', read_local_dataframe)
vineyard.io.read.register('kafka', read_kafka_bytes)
vineyard.io.read.register('kafka', read_kafka_dataframe)
vineyard.io.read.register('hdfs', read_hdfs_dataframe)
vineyard.io.read.register('hive', read_hive_dataframe)
vineyard.io.read.register('vineyard', read_vineyard_dataframe)
vineyard.io.read.register('oss', read_oss_dataframe)


def parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('parse_dataframe_to_bytes'), *((vineyard_socket, dataframe_stream) + args), **kwargs)
    return launcher.wait()


def write_local_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_local_orc'), *((vineyard_socket, dataframe_stream, path) + args), **kwargs)
    launcher.join()


def write_local_bytes(path, byte_stream, vineyard_socket, *args, **kwargs):
    path = json.dumps('file://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_local_bytes'), *((vineyard_socket, byte_stream, path) + args), **kwargs)
    launcher.join()


def write_local_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        write_local_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs)
    else:
        write_local_bytes(path, parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs.copy()),
                          vineyard_socket, *args, **kwargs.copy())


def write_kafka_bytes(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_kafka_bytes'), *((vineyard_socket, path, dataframe_stream) + args), **kwargs)
    launcher.join()


def write_kafka_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_kafka_dataframe'), *((vineyard_socket, path, dataframe_stream) + args), **kwargs)
    launcher.join()


def write_hdfs_bytes(path, byte_stream, vineyard_socket, *args, **kwargs):
    path = json.dumps('hdfs://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_hdfs_bytes'), *((vineyard_socket, byte_stream, path) + args), **kwargs)
    launcher.join()


def write_hdfs_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_hdfs_orc'), *((vineyard_socket, dataframe_stream, 'hdfs://' + path) + args),
                 **kwargs)
    launcher.join()


def write_hdfs_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    if '.orc' in path:
        write_hdfs_orc(path, dataframe_stream, vineyard_socket, *args, **kwargs)
    else:
        write_hdfs_bytes(path, parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs.copy()),
                         vineyard_socket, *args, **kwargs.copy())


def write_vineyard_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_vineyard_dataframe'), *((vineyard_socket, dataframe_stream) + args), **kwargs)
    return launcher.wait_all(name=path)


def write_oss_bytes(path, byte_stream, vineyard_socket, *args, **kwargs):
    path = json.dumps('oss://' + path)
    deployment = kwargs.pop('deployment', 'ssh')
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable('write_oss_bytes'), *((vineyard_socket, byte_stream, path) + args), **kwargs)
    launcher.join()


def write_oss_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    write_oss_bytes(path, parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs.copy()),
                    vineyard_socket, *args, **kwargs.copy())


vineyard.io.write.register('file', write_local_dataframe)
vineyard.io.write.register('kafka', write_kafka_bytes)
vineyard.io.write.register('kafka', write_kafka_dataframe)
vineyard.io.write.register('hdfs', write_hdfs_dataframe)
vineyard.io.write.register('vineyard', write_vineyard_dataframe)
vineyard.io.write.register('oss', write_oss_dataframe)
