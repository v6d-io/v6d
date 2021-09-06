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

import base64
import json
import logging
import os
from vineyard.data.dataframe import make_global_dataframe

import vineyard.io
from vineyard.launcher.launcher import LauncherStatus
from vineyard.launcher.script import ScriptLauncher

from vineyard.core.resolver import default_resolver_context

logger = logging.getLogger("vineyard")
base_path = os.path.abspath(os.path.dirname(__file__))


def parallel_stream_resolver(obj):
    """Return a list of *local* partial streams."""
    meta = obj.meta
    partition_size = int(meta["size_"])
    return [meta.get_member("stream_%d" % i) for i in range(partition_size)]


def global_dataframe_resolver(obj, resolver):
    """Return a list of dataframes."""
    meta = obj.meta
    num = int(meta['partitions_-size'])

    dataframes = []
    orders = []
    for i in range(num):
        df = meta.get_member('partitions_-%d' % i)
        if df.meta.islocal:
            dataframes.append(resolver.run(df))
            orders.append(df.meta["row_batch_index_"])
    if orders != sorted(orders):
        raise ValueError("Bad dataframe orders:", orders)
    return dataframes


default_resolver_context.register("vineyard::ParallelStream", parallel_stream_resolver)
default_resolver_context.register("vineyard::GlobalDataFrame", global_dataframe_resolver)


def _resolve_ssh_script(deployment="ssh"):
    if deployment == "ssh":
        return os.path.join(base_path, "ssh.sh")
    if deployment == "kubernetes":
        return os.path.join(base_path, "kube_ssh.sh")
    raise ValueError('Unknown deployment: "%s"' % deployment)


class StreamLauncher(ScriptLauncher):
    """Launch the job by executing a script."""
    def __init__(self, vineyard_endpoint=None, deployment="ssh"):
        """Launch a job to read as a vineyard stream.

        Parameters
        ----------
        vineyard_endpoint: str
            IPC or RPC endpoint to connect to vineyard. If not specified, vineyard
            will try to discovery vineyardd from the environment variable named
            :code:`VINEYARD_IPC_SOCKET`.
        """
        self.vineyard_endpoint = vineyard_endpoint
        super(StreamLauncher, self).__init__(_resolve_ssh_script(deployment=deployment))

    def wait(self, timeout=None):
        return vineyard.ObjectID(super(StreamLauncher, self).wait(timeout=timeout))


class ParallelStreamLauncher(ScriptLauncher):
    """Launch the job by executing a script, in which `ssh` or `kubectl exec` will
    be used under the hood.
    """
    def __init__(self, deployment="ssh"):
        self.deployment = deployment
        self.vineyard_endpoint = None
        super(ParallelStreamLauncher, self).__init__(_resolve_ssh_script(deployment=deployment))

        self._streams = []
        self._procs = []

    def run(self, *args, **kwargs):
        """Execute a job to read as a vineyard stream or write a vineyard stream to
        external data sink.

        Parameters
        ----------
        vineyard_endpoint: str
            The local IPC or RPC endpoint to connect to vineyard. If not specified,
            vineyard will try to discovery vineyardd from the environment variable
            named :code:`VINEYARD_IPC_SOCKET` and :code:`VINEYARD_RPC_ENDPOINT`.
        """
        kwargs = kwargs.copy()
        self.vineyard_endpoint = kwargs.pop("vineyard_endpoint", None)
        if ":" in self.vineyard_endpoint:
            self.vineyard_endpoint = tuple(self.vineyard_endpoint.split(":"))

        hosts = kwargs.pop("hosts", ["localhost"])
        num_workers = kwargs.pop("num_workers", len(hosts))

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

        messages = []
        for proc in self._procs:
            if proc.status == LauncherStatus.FAILED and \
                    (proc.exit_code is not None and proc.exit_code != 0):
                messages.append("Failed to launch job [%s], exited with %r: %s" %
                                (proc.command, proc.exit_code, ''.join(proc.error_message)))
        if messages:
            raise RuntimeError("Subprocesses failed with the following error: \n%s" % ('\n\n'.join(messages)))

    def dispose(self, desired=True):
        for proc in self._procs:
            proc.dispose()

    def wait(self, timeout=None, func=None):
        partial_ids = []
        for proc in self._procs:
            r = proc.wait(timeout=timeout)
            partial_ids.append(r)
        logger.debug("partial_ids = %s", partial_ids)
        if func is None:
            return self.create_parallel_stream(partial_ids)
        return func(self.vineyard_endpoint, partial_ids)

    def create_parallel_stream(self, partial_ids):
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::ParallelStream'
        meta.set_global(True)
        meta['size_'] = len(partial_ids)
        for idx, partition_id in enumerate(partial_ids):
            meta.add_member("stream_%d" % idx, partition_id)
        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        ret_meta = vineyard_rpc_client.create_metadata(meta)
        vineyard_rpc_client.persist(ret_meta.id)
        return ret_meta.id

    def wait_all(self, func=None, **kwargs):
        results = []
        for proc in self._procs:
            proc.join()
            results.append(proc._result)
        logger.debug("results of wait_all = %s", results)
        if func is None:
            return self.create_global_dataframe(results, **kwargs)
        return func(self.vineyard_endpoint, results, **kwargs)

    def create_global_dataframe(self, results, **kwargs):
        # use the partial_id_matrix and the name in **kwargs
        # to create a global dataframe. Here the name is given in the
        # the input URI path in the
        # form of vineyard://{name_for_the_global_dataframe}
        name = kwargs.pop("name", None)
        if name is None:
            raise ValueError("Name of the global dataframe is not provided")

        chunks = []
        for row in results:
            for chunk in row:
                chunks.append(chunk)

        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        extra_meta = {
            'partition_shape_row_': len(results),
            'partition_shape_column_': 1,
            'nbytes': 0,  # FIXME
        }
        gdf = make_global_dataframe(vineyard_rpc_client, chunks, extra_meta)
        vineyard_rpc_client.put_name(gdf, name)


def get_executable(name):
    return f"vineyard_{name}"


def parse_bytes_to_dataframe(vineyard_socket, byte_stream, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("parse_bytes_to_dataframe"),
        vineyard_socket,
        byte_stream,
        *args,
        **kwargs,
    )
    return launcher.wait()


def read_kafka_bytes(path, vineyard_socket, *args, **kwargs):
    """Read a bytes stream from a kafka topic."""
    path = json.dumps(path)
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable("read_kafka_bytes"), vineyard_socket, path, *args, **kwargs)
    return launcher.wait()


def read_kafka_dataframe(path, vineyard_socket, *args, **kwargs):
    stream = read_kafka_bytes(path, vineyard_socket, *args, **kwargs.copy())
    return parse_bytes_to_dataframe(
        vineyard_socket,
        stream,
        *args,
        **kwargs.copy(),
    )


def read_vineyard_dataframe(path, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    storage_options = kwargs.pop("storage_options", {})
    read_options = kwargs.pop("read_options", {})
    storage_options = base64.b64encode(json.dumps(storage_options).encode("utf-8")).decode("utf-8")
    read_options = base64.b64encode(json.dumps(read_options).encode("utf-8")).decode("utf-8")
    # Note that vineyard currently supports hive tables stored as orc format only
    launcher.run(
        get_executable("read_vineyard_dataframe"),
        vineyard_socket,
        path,
        storage_options,
        read_options,
        *args,
        **kwargs,
    )
    return launcher.wait()


def read_bytes(path, vineyard_socket, storage_options, read_options, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("read_bytes"),
        vineyard_socket,
        path,
        storage_options,
        read_options,
        *args,
        **kwargs,
    )
    return launcher.wait()


def read_orc(path, vineyard_socket, storage_options, read_options, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("read_orc"),
        vineyard_socket,
        path,
        storage_options,
        read_options,
        *args,
        **kwargs,
    )
    return launcher.wait()


def read_dataframe(path, vineyard_socket, *args, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    read_options = kwargs.pop("read_options", {})
    storage_options = base64.b64encode(json.dumps(storage_options).encode("utf-8")).decode("utf-8")
    read_options = base64.b64encode(json.dumps(read_options).encode("utf-8")).decode("utf-8")
    if ".orc" in path:
        logger.debug("Read Orc file from %s.", path)
        return read_orc(path, vineyard_socket, storage_options, read_options, *args, **kwargs.copy())
    else:
        stream = read_bytes(path, vineyard_socket, storage_options, read_options, *args, **kwargs.copy())
        return parse_bytes_to_dataframe(
            vineyard_socket,
            stream,
            *args,
            **kwargs.copy(),
        )


vineyard.io.read.register("file", read_dataframe)
vineyard.io.read.register("hdfs", read_dataframe)
vineyard.io.read.register("hive", read_dataframe)
vineyard.io.read.register("s3", read_dataframe)
vineyard.io.read.register("oss", read_dataframe)

vineyard.io.read.register("kafka", read_kafka_bytes)
vineyard.io.read.register("kafka", read_kafka_dataframe)
vineyard.io.read.register("vineyard", read_vineyard_dataframe)


def parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("parse_dataframe_to_bytes"),
        *((vineyard_socket, dataframe_stream) + args),
        **kwargs,
    )
    return launcher.wait()


def write_bytes(path, byte_stream, vineyard_socket, storage_options, write_options, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_bytes"),
        vineyard_socket,
        path,
        byte_stream,
        storage_options,
        write_options,
        *args,
        **kwargs,
    )
    launcher.join()


def write_orc(path, dataframe_stream, vineyard_socket, storage_options, write_options, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_orc"),
        vineyard_socket,
        path,
        dataframe_stream,
        storage_options,
        write_options,
        *args,
        **kwargs,
    )
    launcher.join()


def write_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    write_options = kwargs.pop("write_options", {})
    storage_options = base64.b64encode(json.dumps(storage_options).encode("utf-8")).decode("utf-8")
    write_options = base64.b64encode(json.dumps(write_options).encode("utf-8")).decode("utf-8")
    if ".orc" in path:
        logger.debug("Write Orc file to %s.", path)
        write_orc(
            path,
            dataframe_stream,
            vineyard_socket,
            storage_options,
            write_options,
            *args,
            **kwargs.copy(),
        )
    else:
        stream = parse_dataframe_to_bytes(vineyard_socket, dataframe_stream, *args, **kwargs.copy())
        write_bytes(path, stream, vineyard_socket, storage_options, write_options, *args, **kwargs.copy())


def write_kafka_bytes(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_kafka_bytes"),
        *((vineyard_socket, path, dataframe_stream) + args),
        **kwargs,
    )
    launcher.join()


def write_kafka_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_kafka_dataframe"),
        *((vineyard_socket, path, dataframe_stream) + args),
        **kwargs,
    )
    launcher.join()


def write_vineyard_dataframe(path, dataframe_stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_vineyard_dataframe"),
        vineyard_socket,
        dataframe_stream,
        *args,
        **kwargs,
    )
    return launcher.wait_all(name=path[len("vineyard://"):])


vineyard.io.write.register("file", write_dataframe)
vineyard.io.write.register("hdfs", write_dataframe)
vineyard.io.write.register("s3", write_dataframe)
vineyard.io.write.register("oss", write_dataframe)

vineyard.io.write.register("kafka", write_kafka_bytes)
vineyard.io.write.register("kafka", write_kafka_dataframe)
vineyard.io.write.register("vineyard", write_vineyard_dataframe)


def serialize_to_stream(object_id, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable("serializer"), vineyard_socket, object_id, *args, **kwargs)
    return launcher.wait()


def serialize(path, object_id, vineyard_socket, *args, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    write_options = kwargs.pop("write_options", {})
    write_options['serialization_mode'] = True
    storage_options = base64.b64encode(json.dumps(storage_options).encode("utf-8")).decode("utf-8")
    write_options = base64.b64encode(json.dumps(write_options).encode("utf-8")).decode("utf-8")

    stream = serialize_to_stream(object_id, vineyard_socket, *args, **kwargs.copy())
    write_bytes(path, stream, vineyard_socket, storage_options, write_options, *args, **kwargs.copy())


vineyard.io.serialize.register("global", serialize)


def deserialize_from_stream(stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(get_executable("deserializer"), vineyard_socket, stream, *args, **kwargs)

    def func(vineyard_endpoint, results):
        # results format:
        # one base64encoded meta string
        # others are ';' separated old_id -> new-id map
        # x1:y1;x2:y2;
        id_map = {}
        meta = {}
        for row in results:
            for column in row:
                if ":" in column:
                    pairs = column.split(";")
                    for pair in pairs:
                        if pair:
                            old_id, new_id = pair.split(":")
                            id_map[old_id] = new_id
                else:
                    meta = base64.b64decode(column.encode("utf-8")).decode("utf-8")
        meta = json.loads(meta)
        new_meta = vineyard.ObjectMeta()
        for key, value in meta.items():
            if isinstance(value, dict):
                new_meta.add_member(key, vineyard.ObjectID(id_map[value['id']]))
            else:
                new_meta[key] = value
        vineyard_rpc_client = vineyard.connect(vineyard_endpoint)
        ret_meta = vineyard_rpc_client.create_metadata(new_meta)
        vineyard_rpc_client.persist(ret_meta)
        return ret_meta.id

    return launcher.wait_all(func=func)


def deserialize(path, vineyard_socket, *args, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    read_options = kwargs.pop("read_options", {})
    read_options['serialization_mode'] = True
    storage_options = base64.b64encode(json.dumps(storage_options).encode("utf-8")).decode("utf-8")
    read_options = base64.b64encode(json.dumps(read_options).encode("utf-8")).decode("utf-8")
    stream = read_bytes(path, vineyard_socket, storage_options, read_options, *args, **kwargs)
    return deserialize_from_stream(stream, vineyard_socket, *args, **kwargs.copy())


vineyard.io.deserialize.register("global", deserialize)
