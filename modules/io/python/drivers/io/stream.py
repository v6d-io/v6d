#! /usr/bin/env python
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
import logging
import os
from typing import Callable
from typing import List
from typing import Union

import vineyard.io
from vineyard._C import Object
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard.core.utils import ReprableString
from vineyard.data.dataframe import make_global_dataframe
from vineyard.launcher.launcher import LauncherStatus
from vineyard.launcher.script import ScriptLauncher

logger = logging.getLogger("vineyard")
base_path = os.path.abspath(os.path.dirname(__file__))


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
            will try to find vineyardd from the environment variable named
            :code:`VINEYARD_IPC_SOCKET`.
        """
        self.vineyard_endpoint = vineyard_endpoint
        super().__init__(_resolve_ssh_script(deployment=deployment))

    def wait(self, timeout=None):
        return vineyard.ObjectID(super().wait(timeout=timeout))


class ParallelStreamLauncher(ScriptLauncher):
    """Launch the job by executing a script, in which `ssh` or `kubectl exec` will
    be used under the hood.
    """

    def __init__(self, deployment="ssh"):
        self.deployment = deployment
        self.vineyard_endpoint = None
        super().__init__(_resolve_ssh_script(deployment=deployment))

        self._streams = []
        self._procs: List[StreamLauncher] = []
        self._stopped = False

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
        if self._stopped:
            return
        self._stopped = True

        for proc in self._procs:
            proc.join()

        messages = []
        for proc in self._procs:
            if proc.status == LauncherStatus.FAILED and (
                proc.exit_code is not None and proc.exit_code != 0
            ):
                if isinstance(proc.command, list):
                    cmd = ' '.join(proc.command)
                else:
                    cmd = proc.command
                messages.append(
                    "Failed to launch job [%s], exited with %r: %s"
                    % (cmd, proc.exit_code, ''.join(proc.error_message))
                )
        if messages:
            raise RuntimeError(
                ReprableString(
                    "Subprocesses failed with the following error: \n%s\n"
                    "extra diagnostics are as follows: %s"
                    % ('\n\n'.join(messages), '\n\n'.join(proc.diagnostics))
                )
            )

    def dispose(self, desired=True):
        if self._stopped:
            return
        self._stopped = True

        for proc in self._procs:
            proc.dispose()

    def wait(  # pylint: disable=arguments-differ
        self,
        timeout=None,
        aggregator: Callable[
            [str, List[ObjectID]], Union[Object, ObjectID, ObjectMeta]
        ] = None,
        **kwargs,
    ):
        '''Wait util the _first_ result on each launcher is ready.'''
        partial_ids = []
        for proc in self._procs:
            r = proc.wait(timeout=timeout)
            partial_ids.append(r)
        logger.debug("[wait] partial ids = %s", partial_ids)
        if aggregator is None:
            return self.create_parallel_stream(partial_ids, **kwargs)
        return aggregator(self.vineyard_endpoint, partial_ids, **kwargs)

    def join_with_aggregator(
        self,
        aggregator: Callable[
            [str, List[List[ObjectID]]], Union[Object, ObjectID, ObjectMeta]
        ],
        **kwargs,
    ):
        """Wait util _all_ results on each launcher is ready and until the launcher
        finishes its work.
        """
        self.join()

        results = []
        for proc in self._procs:
            results.append(proc._result)
        logger.debug("[join_with_aggregator] partial ids = %s", results)
        return aggregator(self.vineyard_endpoint, results, **kwargs)

    def create_parallel_stream(self, partial_ids) -> ObjectID:
        meta = vineyard.ObjectMeta()
        meta['typename'] = 'vineyard::ParallelStream'
        meta.set_global(True)
        meta['__streams_-size'] = len(partial_ids)
        for idx, partition_id in enumerate(partial_ids):
            meta.add_member("__streams_-%d" % idx, partition_id)
        vineyard_rpc_client = vineyard.connect(self.vineyard_endpoint)
        ret_meta = vineyard_rpc_client.create_metadata(meta)
        vineyard_rpc_client.persist(ret_meta.id)
        return ret_meta.id


def get_executable(name):
    return f"vineyard_{name}"


def parse_bytes_to_dataframe(
    vineyard_socket, byte_stream, *args, handlers=None, **kwargs
):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("parse_bytes_to_dataframe"),
        vineyard_socket,
        byte_stream,
        *args,
        **kwargs,
    )
    if handlers is not None:
        handlers.append(launcher)
    return launcher.wait()


def read_vineyard_dataframe(path, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    storage_options = kwargs.pop("storage_options", {})
    read_options = kwargs.pop("read_options", {})
    storage_options = base64.b64encode(
        json.dumps(storage_options).encode("utf-8")
    ).decode("utf-8")
    read_options = base64.b64encode(json.dumps(read_options).encode("utf-8")).decode(
        "utf-8"
    )
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


def read_bytes(
    path, vineyard_socket, storage_options, read_options, *args, handlers=None, **kwargs
):
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
    if handlers is not None:
        handlers.append(launcher)
    return launcher.wait()


def read_orc(
    path, vineyard_socket, storage_options, read_options, *args, handlers=None, **kwargs
):
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
    if handlers is not None:
        handlers.append(launcher)
    return launcher.wait()


def read_dataframe(path, vineyard_socket, *args, handlers=None, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    read_options = kwargs.pop("read_options", {})
    storage_options = base64.b64encode(
        json.dumps(storage_options).encode("utf-8")
    ).decode("utf-8")
    read_options = base64.b64encode(json.dumps(read_options).encode("utf-8")).decode(
        "utf-8"
    )
    if ".orc" in path:
        return read_orc(
            path,
            vineyard_socket,
            storage_options,
            read_options,
            *args,
            handlers=handlers,
            **kwargs.copy(),
        )
    else:
        stream = read_bytes(
            path,
            vineyard_socket,
            storage_options,
            read_options,
            *args,
            handlers=handlers,
            **kwargs.copy(),
        )
        return parse_bytes_to_dataframe(
            vineyard_socket,
            stream,
            *args,
            handlers=handlers,
            **kwargs.copy(),
        )


vineyard.io.read.register("file", read_dataframe)
vineyard.io.read.register("hdfs", read_dataframe)
vineyard.io.read.register("hive", read_dataframe)
vineyard.io.read.register("s3", read_dataframe)
vineyard.io.read.register("oss", read_dataframe)

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


def write_bytes(
    path, byte_stream, vineyard_socket, storage_options, write_options, *args, **kwargs
):
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


def write_orc(
    path,
    dataframe_stream,
    vineyard_socket,
    storage_options,
    write_options,
    *args,
    **kwargs,
):
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
    storage_options = base64.b64encode(
        json.dumps(storage_options).encode("utf-8")
    ).decode("utf-8")
    write_options = base64.b64encode(json.dumps(write_options).encode("utf-8")).decode(
        "utf-8"
    )
    if ".orc" in path:
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
        stream = parse_dataframe_to_bytes(
            vineyard_socket, dataframe_stream, *args, **kwargs.copy()
        )
        write_bytes(
            path,
            stream,
            vineyard_socket,
            storage_options,
            write_options,
            *args,
            **kwargs.copy(),
        )


def create_global_dataframe(
    vineyard_endpoint: str, results: List[List[ObjectID]], name: str, **_kwargs
) -> ObjectID:
    # use the partial_id_matrix and the name in **kwargs to create a global
    # dataframe.
    #
    # Here the `name`` is given in the the input URI path in the form of
    # vineyard://{name_for_the_global_dataframe}
    if name is None:
        raise ValueError("Name of the global dataframe is not provided")

    chunks = []
    for subresults in results:
        chunks.extend(subresults)

    vineyard_rpc_client = vineyard.connect(vineyard_endpoint)
    extra_meta = {
        'partition_shape_row_': len(results),
        'partition_shape_column_': 1,
        'nbytes': 0,  # FIXME
    }
    gdf = make_global_dataframe(vineyard_rpc_client, chunks, extra_meta)
    vineyard_rpc_client.put_name(gdf, name)
    vineyard_rpc_client.persist(gdf.id)
    return gdf.id


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
    return launcher.join_with_aggregator(
        aggregator=create_global_dataframe, name=path[len("vineyard://") :]
    )


vineyard.io.write.register("file", write_dataframe)
vineyard.io.write.register("hdfs", write_dataframe)
vineyard.io.write.register("s3", write_dataframe)
vineyard.io.write.register("oss", write_dataframe)

vineyard.io.write.register("vineyard", write_vineyard_dataframe)


def merge_global_object(vineyard_endpoint, results: List[List[ObjectID]]) -> ObjectID:
    if results is None or len(results) == 0:
        raise ValueError("No available sub objects to merge")

    chunks = []
    for subresults in results:
        chunks.extend(subresults)

    if len(chunks) == 0:
        raise ValueError("No available sub objects to merge")

    if len(chunks) == 1:
        # fastpath: no need to merge
        if not isinstance(chunks[0], ObjectID):
            return ObjectID(chunks[0])
        else:
            return chunks[0]

    vineyard_rpc_client = vineyard.connect(vineyard_endpoint)
    metadatas = []
    for chunk in chunks:
        if not isinstance(chunk, ObjectID):
            chunk = ObjectID(chunk)
        metadatas.append(vineyard_rpc_client.get_meta(chunk))

    chunkmap, isglobal = dict(), False
    for meta in metadatas:
        if meta.isglobal:
            isglobal = True
            for k, v in meta.items():
                if isinstance(v, ObjectMeta):
                    chunkmap[v.id] = k
        else:
            if isglobal:
                raise ValueError('Not all sub objects are global objects: %s' % results)

    if not isglobal:
        raise ValueError(
            "Unable to merge more than one non-global objects: %s" % results
        )

    base_meta = ObjectMeta()
    base_meta.set_global(True)
    for k, v in metadatas[0].items():
        if isinstance(v, ObjectMeta):
            continue
        if k in ['id', 'signature', 'instance_id']:
            continue
        base_meta[k] = v
    for v, k in chunkmap.items():
        base_meta.add_member(k, v)
    meta = vineyard_rpc_client.create_metadata(base_meta)
    vineyard_rpc_client.persist(meta.id)
    return meta.id


def write_bytes_collection(
    path, byte_stream, vineyard_socket, storage_options, *args, **kwargs
):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("write_bytes_collection"),
        vineyard_socket,
        path,
        byte_stream,
        storage_options,
        *args,
        **kwargs,
    )
    launcher.join()


def read_bytes_collection(path, vineyard_socket, storage_options, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("read_bytes_collection"),
        vineyard_socket,
        path,
        storage_options,
        *args,
        **kwargs,
    )
    return launcher.wait()


def serialize_to_stream(object_id, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    serialization_options = kwargs.pop("serialization_options", {})
    serialization_options = base64.b64encode(
        json.dumps(serialization_options).encode("utf-8")
    ).decode("utf-8")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("serializer"),
        vineyard_socket,
        object_id,
        serialization_options,
        *args,
        **kwargs,
    )
    return launcher.wait()


def serialize(path, object_id, vineyard_socket, *args, **kwargs):
    path = json.dumps(path)
    storage_options = kwargs.pop("storage_options", {})
    storage_options = base64.b64encode(
        json.dumps(storage_options).encode("utf-8")
    ).decode("utf-8")

    stream = serialize_to_stream(object_id, vineyard_socket, *args, **kwargs.copy())
    write_bytes_collection(
        path, stream, vineyard_socket, storage_options, *args, **kwargs.copy()
    )


def deserialize_from_stream(stream, vineyard_socket, *args, **kwargs):
    deployment = kwargs.pop("deployment", "ssh")
    launcher = ParallelStreamLauncher(deployment)
    launcher.run(
        get_executable("deserializer"),
        vineyard_socket,
        stream,
        *args,
        **kwargs,
    )
    return launcher.join_with_aggregator(aggregator=merge_global_object)


def deserialize(path, vineyard_socket, *args, **kwargs):
    storage_options = kwargs.pop("storage_options", {})
    storage_options = base64.b64encode(
        json.dumps(storage_options).encode("utf-8")
    ).decode("utf-8")
    stream = read_bytes_collection(
        path, vineyard_socket, storage_options, *args, **kwargs
    )
    return deserialize_from_stream(stream, vineyard_socket, *args, **kwargs.copy())


vineyard.io.serialize.register("default", serialize)
vineyard.io.deserialize.register("default", deserialize)

# for backwards compatibility
vineyard.io.serialize.register("global", serialize)
vineyard.io.deserialize.register("global", deserialize)
