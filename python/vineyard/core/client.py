#! /usr/bin/env python
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

import contextlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from vineyard import envvars
from vineyard._C import Blob
from vineyard._C import BlobBuilder
from vineyard._C import IPCClient
from vineyard._C import NotEnoughMemoryException
from vineyard._C import Object
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard._C import RemoteBlob
from vineyard._C import RemoteBlobBuilder
from vineyard._C import RPCClient
from vineyard._C import VineyardException
from vineyard._C import _connect
from vineyard.core.builder import BuilderContext
from vineyard.core.builder import get_current_builders
from vineyard.core.builder import put
from vineyard.core.resolver import ResolverContext
from vineyard.core.resolver import get


def _apply_docstring(func):
    def _apply(fn):
        fn.__doc__ = func.__doc__
        return fn

    return _apply


def _parse_configuration(config) -> Tuple[Optional[str], Optional[str]]:
    '''Parse vineyard IPC socket and RPC endpoints from configuration.

    Parameters:
        config: Path to a YAML configuration file or a directory containing
                the default config file `vineyard-config.yaml`. If you define
                the directory of vineyard config, the
                `/directory/vineyard-config.yaml` and
                `/directory/vineyard/vineyard-config.yaml` will be parsed.

    Returns:
        (socket, endpoints): IPC socket path and RPC endpoints.
    '''
    if not config:
        return None, None

    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None, None

    if os.path.isdir(config):
        config_options = ['vineyard-config.yaml', 'vineyard/vineyard-config.yaml']
        for config_option in config_options:
            config_path = os.path.join(config, config_option)
            if os.path.isfile(config_path):
                config = config_path
                break
    if not os.path.isfile(config):
        return None, None

    try:
        with open(config, 'r', encoding='utf-8') as f:
            vineyard_config = yaml.safe_load(f).get('Vineyard', {})
    except:  # noqa: E722, pylint: disable=bare-except
        return None, None

    ipc_socket = vineyard_config.get('IPCSocket', None)
    rpc_endpoint = vineyard_config.get('RPCEndpoint', None)

    if ipc_socket:
        if not os.path.isabs(ipc_socket):
            base_dir = os.path.dirname(config) if os.path.isfile(config) else config
            ipc_socket = os.path.join(base_dir, ipc_socket)

        if not os.path.exists(ipc_socket):
            ipc_socket = None

    return ipc_socket, rpc_endpoint


def _is_blob(object_id: ObjectID):
    """_is_blob_

    Args:
        object_id (ObjectID): ObjectID to check if it is a blob

    Returns:
        bool: True if the object_id is a blob, False otherwise
    """
    return int(object_id) & 0x8000000000000000


def _traverse_blobs(meta: ObjectMeta, blobs=None):
    """_traverse_blobs_

    Recursively traverses ObjectMeta to find and accumulate blob IDs by instance_id.

    Args:
        meta (ObjectMeta): ObjectMeta to traverse for blobs.
        blobs (dict, optional): Accumulator for blobs organized by instance_id.

    Returns:
        dict: A dictionary of blobs organized by instance_id.
    """

    if blobs is None:
        blobs = {}

    def add_blob(instance_id, blob_id):
        if instance_id not in blobs:
            blobs[instance_id] = []
        blobs[instance_id].append(blob_id)

    if _is_blob(meta.id):
        add_blob(meta.instance_id, meta.id)
    else:
        for _, v in meta.items():
            if isinstance(v, ObjectMeta):
                if _is_blob(v.id):
                    add_blob(v.instance_id, v.id)
                else:
                    _traverse_blobs(v, blobs)

    return blobs


class Client:
    """Client is responsible for managing IPC and RPC clients for Vineyard
    and provides a high-level interface to fetch objects from the Vineyard cluster.
    """

    def __init__(
        self,
        socket: str = None,
        port: Union[int, str] = None,
        # move host after port to make sure unnamed (host, port) works
        host: str = None,
        endpoint: Tuple[str, Union[str, int]] = None,
        rdma_endpoint: Union[str, Tuple[str, Union[str, int]]] = None,
        session: int = None,
        username: str = None,
        password: str = None,
        max_workers: int = 8,
        config: str = None,
    ):
        """Connects to the vineyard IPC socket and RPC socket.

        - For the IPC Client, the argument `socket` takes precedence over the
          environment variable `VINEYARD_IPC_SOCKET`, which in turn takes precedence
          over the `IPCSocket` field in the config file."
        - For the RPC Client, the argument `endpoint` takes precedence over the
          argument `host` and `port`, which in turn takes precedence over the
          environment variable `VINEYARD_RPC_ENDPOINT`, which further takes precedence
          over the `RPCEndpoint` field in the config file. Besides, if you have the
          eRDMA or RDMA hardware, you can define the environment variable
          `VINEYARD_RDMA_ENDPOINT` to specify the RDMA endpoint to speed up the data
          transfer.

        The `connect()` API can be used in following ways:

        - `connect()` without any arguments, which will try to connect to the vineyard
          by resolving endpoints from the environment variables.
        - `connect('/path/to/vineyard.sock')`, which will try to establish an IPC
          connection.
        - `connect('hostname:port')`, which will try to establish an RPC connection.
        - `connect('hostname', port)`, which will try to establish an RPC connection.
        - `connect(endpoint=('hostname', port))`, which will try to establish an RPC
          connection.
        - `connect(endpoint=('hostname', port), rdma_endpoint=('rdma_host', port))`,
          which will try to establish an RPC connection with RDMA enabled.
        - `connect(config='/path/to/vineyard-config.yaml')`, which will try to
          resolve the IPC socket and RPC endpoints from the configuration file.

        Parameters:
            socket: Optional, the path to the IPC socket, or RPC endpoints of format
                    `host:port`.
            port: Optional, the port of the RPC endpoint.
            host: Optional, the host of the RPC endpoint.
            endpoint: Optional, the RPC endpoint of format `host:port`.
            rdma_endpoint: Optional, the RDMA endpoint of format `host:port` or
                      ('host', port) or ('host', 'port')
            session: Optional, the session id to connect.
            username: Optional, the required username of vineyardd when authentication
                      is enabled.
            password: Optional, the required password of vineyardd when authentication
                      is enabled.
            max_workers: Optional, the maximum number of threads that can be used to
                        asynchronously put objects to vineyard. Default is 8.
            config: Optional, can either be a path to a YAML configuration file or
                    a path to a directory containing the default config file
                    `vineyard-config.yaml`. Also, the environment variable
                    `VINEYARD_CONFIG` can be used to specify the
                    path to the configuration file. If not defined, the default
                    config file `/var/run/vineyard-config.yaml` or
                    `/var/run/vineyard/vineyard-config.yaml`

        The content of the configuration file should has the following content:

        .. code:: yaml

            Vineyard:
                IPCSocket: '/path/to/vineyard.sock'
                RPCEndpoint: 'hostname1:port1,hostname2:port2,...'
        """
        self._ipc_client: IPCClient = None
        self._rpc_client: RPCClient = None

        kwargs = {}
        if session is not None:
            kwargs['session'] = session
        if username is not None:
            kwargs['username'] = username
        if password is not None:
            kwargs['password'] = password

        if socket is not None and port is not None and host is None:
            socket, host = None, socket

        hosts, ports = [], []
        if not socket:
            socket = os.getenv('VINEYARD_IPC_SOCKET', None)
        if not endpoint and not (host and port):
            endpoint = os.getenv('VINEYARD_RPC_ENDPOINT', None)
        if not config:
            config = os.getenv('VINEYARD_CONFIG', '/var/run')
        if endpoint:
            if not isinstance(endpoint, (tuple, list)):
                for ep in endpoint.split(','):
                    h, p = [e.strip() for e in ep.split(':')]
                    hosts.append(h)
                    ports.append(p)
            else:
                h, p = endpoint
                hosts = [h]
                ports = [p]

        if host and port:
            hosts.append(host)
            ports.append(port)

        if rdma_endpoint:
            if isinstance(rdma_endpoint, (tuple)):
                rdma_endpoint = ':'.join(map(str, rdma_endpoint))
        else:
            rdma_endpoint = os.getenv('VINEYARD_RDMA_ENDPOINT', '')

        if config and ((not socket) or (not (hosts and ports))):
            ipc_socket, rpc_endpoint = _parse_configuration(config)
            if ipc_socket and not socket:
                socket = ipc_socket
            if rpc_endpoint and not (hosts and ports):
                for ep in rpc_endpoint.split(','):
                    h, p = [e.strip() for e in ep.split(':')]
                    hosts.append(h)
                    ports.append(p)

        if socket:
            self._ipc_client = _connect(socket, **kwargs)
        for host, port in zip(hosts, ports):
            try:
                self._rpc_client = _connect(
                    host, port, rdma_endpoint=rdma_endpoint, **kwargs
                )
                break
            except VineyardException:
                continue

        self._max_workers = max_workers
        self._put_thread_pool = None

        self._spread = False
        self._compression = True
        if self._ipc_client is None and self._rpc_client is None:
            raise ConnectionError(
                "Failed to connect to vineyard via both IPC and RPC connection. "
                "Arguments, environment variables `VINEYARD_IPC_SOCKET` "
                "and `VINEYARD_RPC_ENDPOINT`, as well as the configuration file, "
                "are all unavailable."
            )

    @property
    def compression(self) -> bool:
        '''Whether the compression is enabled for underlying RPC client.'''
        if self._rpc_client:
            return self._rpc_client.compression
        return self._compression

    @compression.setter
    def compression(self, value: bool = True):
        if self._rpc_client:
            self._rpc_client.compression = value
        self._compression = value

    @property
    def spread(self) -> bool:
        '''Whether the spread is enabled for underlying RPC client.'''
        return self._spread

    @spread.setter
    def spread(self, value: bool = False):
        self._spread = value

    @property
    def timeout_seconds(self) -> int:
        """
        The timeout seconds for underlying client.
        Default is 300 seconds.
        """
        return self.default_client().timeout_seconds

    @timeout_seconds.setter
    def timeout_seconds(self, value: int):
        if self._ipc_client:
            self._ipc_client.timeout_seconds = value
        if self._rpc_client:
            self._rpc_client.timeout_seconds = value

    @property
    def ipc_client(self) -> IPCClient:
        assert self._ipc_client is not None, "IPC client is not available."
        return self._ipc_client

    @property
    def rpc_client(self) -> RPCClient:
        assert self._rpc_client is not None, "RPC client is not available."
        return self._rpc_client

    @property
    def put_thread_pool(self) -> ThreadPoolExecutor:
        """Lazy initialization of the thread pool for asynchronous put."""
        if self._put_thread_pool is None:
            self._put_thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._put_thread_pool

    def has_ipc_client(self):
        return self._ipc_client is not None

    def has_rpc_client(self):
        return self._rpc_client is not None

    def default_client(self) -> Union[IPCClient, RPCClient]:
        return self._ipc_client if self._ipc_client else self._rpc_client

    # The following functions are wrappers of the corresponding functions in the
    # ClientBase class.

    @_apply_docstring(IPCClient.create_metadata)
    def create_metadata(
        self, metadata: Union[ObjectMeta, List[ObjectMeta]], instance_id: int = None
    ) -> Union[ObjectMeta, List[ObjectMeta]]:
        if instance_id is not None:
            return self.default_client().create_metadata(metadata, instance_id)
        return self.default_client().create_metadata(metadata)

    @_apply_docstring(IPCClient.delete)
    def delete(
        self,
        object: Union[ObjectID, Object, ObjectMeta, List[ObjectID]],
        force: bool = False,
        deep: bool = True,
        memory_trim: bool = False,
    ) -> None:
        return self.default_client().delete(object, force, deep, memory_trim)

    @_apply_docstring(IPCClient.create_stream)
    def create_stream(self, id: ObjectID) -> None:
        return self.default_client().create_stream(id)

    @_apply_docstring(IPCClient.open_stream)
    def open_stream(self, id: ObjectID, mode: str) -> None:
        return self.default_client().open_stream(id, mode)

    @_apply_docstring(IPCClient.push_chunk)
    def push_chunk(self, stream_id: ObjectID, chunk: ObjectID) -> None:
        return self.default_client().push_chunk(stream_id, chunk)

    @_apply_docstring(IPCClient.next_chunk_id)
    def next_chunk_id(self, stream_id: ObjectID) -> ObjectID:
        return self.default_client().next_chunk_id(stream_id)

    @_apply_docstring(IPCClient.next_chunk_meta)
    def next_chunk_meta(self, stream_id: ObjectID) -> ObjectMeta:
        return self.default_client().next_chunk_meta(stream_id)

    @_apply_docstring(IPCClient.next_chunk)
    def next_chunk(self, stream_id: ObjectID) -> Object:
        return self.default_client().next_chunk(stream_id)

    @_apply_docstring(IPCClient.stop_stream)
    def stop_stream(self, stream_id: ObjectID, failed: bool) -> None:
        return self.default_client().stop_stream(stream_id, failed)

    @_apply_docstring(IPCClient.drop_stream)
    def drop_stream(self, stream_id: ObjectID) -> None:
        return self.default_client().drop_stream(stream_id)

    @_apply_docstring(IPCClient.persist)
    def persist(self, object: Union[ObjectID, Object, ObjectMeta]) -> None:
        return self.default_client().persist(object)

    @_apply_docstring(IPCClient.exists)
    def exists(self, object: ObjectID) -> bool:
        return self.default_client().exists(object)

    @_apply_docstring(IPCClient.shallow_copy)
    def shallow_copy(
        self, object_id: ObjectID, extra_metadata: dict = None
    ) -> ObjectID:
        if extra_metadata:
            return self.default_client().shallow_copy(object_id, extra_metadata)
        return self.default_client().shallow_copy(object_id)

    @_apply_docstring(IPCClient.list_names)
    def list_names(
        self, pattern: str, regex: bool = False, limit: int = 5
    ) -> List[str]:
        return self.default_client().list_names(pattern, regex, limit)

    @_apply_docstring(IPCClient.put_name)
    def put_name(self, object: Union[Object, ObjectMeta, ObjectID], name: str) -> None:
        return self.default_client().put_name(object, name)

    @_apply_docstring(IPCClient.get_name)
    def get_name(self, name: str, wait: bool = False) -> ObjectID:
        return self.default_client().get_name(name, wait)

    @_apply_docstring(IPCClient.drop_name)
    def drop_name(self, name: str) -> None:
        return self.default_client().drop_name(name)

    @_apply_docstring(IPCClient.sync_meta)
    def sync_meta(self) -> None:
        return self.default_client().sync_meta()

    @_apply_docstring(IPCClient.migrate)
    def migrate(self, object_id: ObjectID) -> ObjectID:
        return self.default_client().migrate(object_id)

    @_apply_docstring(IPCClient.clear)
    def clear(self) -> None:
        return self.default_client().clear()

    @_apply_docstring(IPCClient.memory_trim)
    def memory_trim(self) -> bool:
        return self.default_client().memory_trim()

    @_apply_docstring(IPCClient.label)
    def label(
        self,
        object_id: ObjectID,
        key_or_labels: Union[str, Dict[str, str]],
        value: str = None,
    ) -> None:
        if isinstance(key_or_labels, dict) and value is None:
            return self.default_client().label(object_id, key_or_labels)
        else:
            return self.default_client().label(object_id, key_or_labels, value)

    @_apply_docstring(IPCClient.evict)
    def evict(self, objects: List[ObjectID]) -> None:
        return self.default_client().evict(objects)

    @_apply_docstring(IPCClient.load)
    def load(self, objects: List[ObjectID], pin: bool = False) -> None:
        return self.default_client().load(objects, pin)

    @_apply_docstring(IPCClient.unpin)
    def unpin(self, objects: List[ObjectID]) -> None:
        return self.default_client().unpin(objects)

    @_apply_docstring(IPCClient.reset)
    def reset(self) -> None:
        if self._ipc_client:
            self._ipc_client.reset()
        if self._rpc_client:
            self._rpc_client.reset()

    @property
    @_apply_docstring(IPCClient.connected)
    def connected(self):
        return self.default_client().connected

    @property
    @_apply_docstring(IPCClient.instance_id)
    def instance_id(self):
        return self.default_client().instance_id

    @property
    @_apply_docstring(IPCClient.meta)
    def meta(self):
        return self.default_client().meta

    @property
    @_apply_docstring(IPCClient.status)
    def status(self):
        return self.default_client().status

    @_apply_docstring(IPCClient.debug)
    def debug(self, debug: dict):
        return self.default_client().debug(debug)

    @property
    @_apply_docstring(IPCClient.ipc_socket)
    def ipc_socket(self):
        return self.default_client().ipc_socket

    @property
    @_apply_docstring(IPCClient.rpc_endpoint)
    def rpc_endpoint(self):
        if self._rpc_client:
            return self._rpc_client.rpc_endpoint
        return self.default_client().rpc_endpoint

    @property
    def rdma_endpoint(self):
        if self._rpc_client:
            return self._rpc_client.rdma_endpoint
        return ""

    @property
    @_apply_docstring(IPCClient.is_ipc)
    def is_ipc(self):
        return self.default_client().is_ipc

    @property
    @_apply_docstring(IPCClient.is_rpc)
    def is_rpc(self):
        return self.default_client().is_rpc

    @property
    @_apply_docstring(IPCClient.version)
    def version(self):
        return self.default_client().version

    # The following functions are wrappers of the corresponding functions in the
    # IPCClient and RPCClient classes.

    @_apply_docstring(IPCClient.create_blob)
    def create_blob(
        self, size: Union[int, List[int]]
    ) -> Union[BlobBuilder, List[BlobBuilder]]:
        return self.ipc_client.create_blob(size)

    @_apply_docstring(IPCClient.create_empty_blob)
    def create_empty_blob(self) -> BlobBuilder:
        return self.ipc_client.create_empty_blob()

    @_apply_docstring(IPCClient.get_blob)
    def get_blob(self, object_id: ObjectID, unsafe: bool = False) -> Blob:
        return self.ipc_client.get_blob(object_id, unsafe)

    @_apply_docstring(IPCClient.get_blobs)
    def get_blobs(self, object_ids: List[ObjectID], unsafe: bool = False) -> List[Blob]:
        return self.ipc_client.get_blobs(object_ids, unsafe)

    @_apply_docstring(RPCClient.create_remote_blob)
    def create_remote_blob(
        self, blob_builder: Union[RemoteBlobBuilder, List[RemoteBlobBuilder]]
    ) -> Union[ObjectMeta, List[ObjectMeta]]:
        return self.rpc_client.create_remote_blob(blob_builder)

    @_apply_docstring(RPCClient.get_remote_blob)
    def get_remote_blob(self, object_id: ObjectID, unsafe: bool = False) -> RemoteBlob:
        return self.rpc_client.get_remote_blob(object_id, unsafe)

    @_apply_docstring(RPCClient.get_remote_blobs)
    def get_remote_blobs(
        self, object_ids: List[ObjectID], unsafe: bool = False
    ) -> List[RemoteBlob]:
        return self.rpc_client.get_remote_blobs(object_ids, unsafe)

    @_apply_docstring(IPCClient.get_object)
    def get_object(
        self, object_id: ObjectID, sync_remote: bool = True, fetch: bool = False
    ) -> Object:
        """
        Fetches the object associated with the given object_id from Vineyard.
        The IPC client is preferred if it's available, otherwise the RPC client
        """
        return self._fetch_object(
            object_id, sync_remote=sync_remote, enable_migrate=fetch
        )

    @_apply_docstring(IPCClient.get_objects)
    def get_objects(
        self, object_ids: List[ObjectID], sync_remote: bool = True
    ) -> List[Object]:
        objects = []
        for object_id in object_ids:
            objects.append(self.get_object(object_id, sync_remote))
        return objects

    @_apply_docstring(IPCClient.get_meta)
    def get_meta(
        self,
        object_id: ObjectID,
        sync_remote: bool = True,
    ) -> ObjectMeta:
        return self.default_client().get_meta(object_id, sync_remote)

    @_apply_docstring(IPCClient.get_metas)
    def get_metas(
        self, object_ids: List[ObjectID], sync_remote: bool = True
    ) -> List[ObjectMeta]:
        metas = []
        for object_id in object_ids:
            metas.append(self.get_meta(object_id, sync_remote))
        return metas

    @_apply_docstring(IPCClient.release_object)
    def release_object(self, object_id: ObjectID) -> None:
        if self.has_ipc_client():
            self._ipc_client.release_object(object_id)

    @_apply_docstring(IPCClient.release_objects)
    def release_objects(self, object_ids: List[ObjectID]) -> None:
        if self.has_ipc_client():
            self._ipc_client.release_objects(object_ids)

    @_apply_docstring(IPCClient.list_objects)
    def list_objects(
        self, pattern: str, regex: bool = False, limit: int = 5
    ) -> List[ObjectID]:
        return self.default_client().list_objects(pattern, regex, limit)

    @_apply_docstring(IPCClient.list_metadatas)
    def list_metadatas(
        self, pattern: str, regex: bool = False, limit: int = 5, nobuffer: bool = False
    ) -> List[ObjectMeta]:
        return self.default_client().list_metadatas(pattern, regex, limit, nobuffer)

    @_apply_docstring(IPCClient.new_buffer_chunk)
    def new_buffer_chunk(self, stream: ObjectID, size: int) -> memoryview:
        return self.ipc_client.new_buffer_chunk(stream, size)

    @_apply_docstring(IPCClient.next_buffer_chunk)
    def next_buffer_chunk(self, stream: ObjectID) -> memoryview:
        return self.ipc_client.next_buffer_chunk(stream)

    @_apply_docstring(IPCClient.allocated_size)
    def allocated_size(self, object_id: Union[Object, ObjectID]) -> int:
        return self.ipc_client.allocated_size(object_id)

    @_apply_docstring(IPCClient.is_shared_memory)
    def is_shared_memory(self, pointer: int) -> bool:
        return self.ipc_client.is_shared_memory(pointer)

    @_apply_docstring(IPCClient.find_shared_memory)
    def find_shared_memory(self, pointer: int) -> ObjectID:
        return self.ipc_client.find_shared_memory(pointer)

    @property
    @_apply_docstring(RPCClient.remote_instance_id)
    def remote_instance_id(self) -> int:
        return self.rpc_client.remote_instance_id

    @_apply_docstring(IPCClient.close)
    def close(self) -> None:
        if self._ipc_client:
            self._ipc_client.close()
        if self._rpc_client:
            self._rpc_client.close()

    @_apply_docstring(IPCClient.fork)
    def fork(self) -> 'Client':
        if self._ipc_client:
            self._ipc_client = self._ipc_client.fork()
        if self._rpc_client:
            self._rpc_client = self._rpc_client.fork()
        return self

    def _fetch_object(
        self, object_id: ObjectID, enable_migrate: bool, sync_remote: bool = True
    ) -> Object:
        meta = self.get_meta(object_id, sync_remote=sync_remote)

        if self.has_ipc_client() and enable_migrate:
            # no need to sync remote metadata as the metadata is already fetched
            return self._ipc_client.get_object(object_id, fetch=True, sync_remote=False)

        blobs = _traverse_blobs(meta)

        # If the object is local, return it directly
        if blobs.keys() == {self.instance_id}:
            return Object.from_(meta)

        cluster_info = self.default_client().meta
        meta.force_local()
        meta._client = None

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self._fetch_blobs_from_instance,
                    cluster_info,
                    instance_id,
                    blobs[instance_id],
                    self.compression,
                    self.rdma_endpoint,
                )
                for instance_id in blobs
                if instance_id != self.instance_id
            }

            for future in as_completed(futures):
                fetched_blobs = future.result()
                for blob in fetched_blobs:
                    meta.add_remote_blob(blob)

        return Object.from_(meta)

    def _fetch_blobs_from_instance(
        self, cluster_info, instance_id, blob_ids, compression, rdma_endpoint
    ) -> Object:
        """Fetches all blobs from a given instance id in the Vineyard cluster.

        Args:
            cluster_info (Dict): The cluster information of the Vineyard cluster.
            instance_id (int): The instance id to fetch blobs from.
            blob_ids (List): The list of blob ids to fetch.
            compression (bool): Whether to enable compression for RPC Client.
            rdma_endpoint (str): The RDMA endpoint to use for the RPC Client.

        Returns:
            RemoteBlob(List): The list of fetched remote blobs.
        """
        instance_status = cluster_info.get(instance_id)

        if instance_status is None or instance_status['rpc_endpoint'] is None:
            raise RuntimeError(
                "The rpc endpoint of the vineyard instance "
                f"{instance_id} is not available."
            )

        if self.has_rpc_client() and self.remote_instance_id == instance_id:
            remote_client = self._rpc_client
        else:
            host, port = instance_status['rpc_endpoint'].split(':')
            try:
                with envvars('VINEYARD_RPC_SKIP_RETRY', '1'):
                    remote_client = _connect(host, port, rdma_endpoint=rdma_endpoint)
                    remote_client.compression = compression
            except Exception as exec:
                raise RuntimeError(
                    f"Failed to connect to the vineyard instance {instance_id} "
                    f"at {host}:{port}."
                ) from exec

        return remote_client.get_remote_blobs(blob_ids)

    def _connect_and_get_memory(self, instance_id, rpc_endpoint):
        host, port = rpc_endpoint.split(':')
        try:
            new_client = _connect(host, port)
            current_available_memory = (
                new_client.status.memory_limit - new_client.status.memory_usage
            )
            return instance_id, current_available_memory
        except Exception:
            return instance_id, float('-inf')

    def _find_the_most_available_memory_instance(self) -> int:
        """
        Find the vineyard instance with the most available memory.

        Returns:
            int: The instance id with the most available memory.
            if only have one instance, return -1
        """
        futures = []
        cluster_info = self.default_client().meta
        with ThreadPoolExecutor() as executor:
            for instance_id, status in cluster_info.items():
                if not (
                    status['ipc_socket'] == self.ipc_socket
                    and status['rpc_endpoint'] == self.rpc_endpoint
                ):
                    futures.append(
                        executor.submit(
                            self._connect_and_get_memory,
                            instance_id,
                            status['rpc_endpoint'],
                        )
                    )

        instance_id_with_most_available_memory = -1
        available_memory = float('-inf')

        for future in as_completed(futures):
            instance_id, current_available_memory = future.result()
            if current_available_memory > available_memory:
                instance_id_with_most_available_memory = instance_id
                available_memory = current_available_memory

        return instance_id_with_most_available_memory

    @_apply_docstring(get)
    def get(
        self,
        object_id: Optional[ObjectID] = None,
        name: Optional[str] = None,
        resolver: Optional[ResolverContext] = None,
        fetch: bool = False,
        **kwargs,
    ):
        return get(self, object_id, name, resolver, fetch, **kwargs)

    def _put_internal(
        self,
        value: Any,
        builder: Optional[BuilderContext] = None,
        persist: bool = False,
        name: Optional[str] = None,
        as_async: bool = False,
        **kwargs,
    ):
        try:
            return put(self, value, builder, persist, name, as_async, **kwargs)
        except NotEnoughMemoryException as exec:
            with envvars(
                {'VINEYARD_RPC_SKIP_RETRY': '1', 'VINEYARD_IPC_SKIP_RETRY': '1'}
            ):
                instance_id = self._find_the_most_available_memory_instance()
                previous_compression_state = self.compression
                if instance_id == -1:
                    warnings.warn("No other vineyard instance available")
                    raise exec
                else:
                    meta = self.default_client().meta
                    warnings.warn(
                        f"Put object to the vineyard instance {instance_id}"
                        "with the most available memory."
                    )
                    # connect to the instance with the most available memory
                    self._ipc_client = None
                    if os.path.exists(meta[instance_id]['ipc_socket']):
                        self._ipc_client = _connect(meta[instance_id]['ipc_socket'])
                        # avoid the case the vineyard instance is restarted
                        if self._ipc_client.instance_id != instance_id:
                            self._ipc_client = None
                    host, port = meta[instance_id]['rpc_endpoint'].split(':')
                    self._rpc_client = _connect(host, port)
                    self.compression = previous_compression_state
            return put(self, value, builder, persist, name, as_async, **kwargs)

    @_apply_docstring(put)
    def put(
        self,
        value: Any,
        builder: Optional[BuilderContext] = None,
        persist: bool = False,
        name: Optional[str] = None,
        as_async: bool = False,
        **kwargs,
    ):
        if as_async:

            def _default_callback(future):
                try:
                    result = future.result()
                    if isinstance(result, ObjectID):
                        print(f"Successfully put object {result}", flush=True)
                    elif isinstance(result, ObjectMeta):
                        print(f"Successfully put object {result.id}", flush=True)
                except Exception as e:
                    print(f"Failed to put object: {e}", flush=True)

            current_builder = builder or get_current_builders()

            thread_pool = self.put_thread_pool
            result = thread_pool.submit(
                self._put_internal,
                value,
                current_builder,
                persist,
                name,
                as_async=True,
                **kwargs,
            )
            result.add_done_callback(_default_callback)
            return result
        return self._put_internal(value, builder, persist, name, **kwargs)

    @contextlib.contextmanager
    def with_compression(self, enabled: bool = True):
        """Disable compression for the following put operations."""
        compression = self.compression
        self.compression = enabled
        yield
        self.compression = compression

    @contextlib.contextmanager
    def with_spread(self, enabled: bool = True):
        """Enable spread for the following put operations."""
        tmp_spread = self._spread
        self.spread = enabled
        yield
        self.spread = tmp_spread


__all__ = ['Client']
