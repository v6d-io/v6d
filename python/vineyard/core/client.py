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

import os
import warnings
from typing import Dict
from typing import List
from typing import Union

from vineyard._C import Blob
from vineyard._C import BlobBuilder
from vineyard._C import Object
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard._C import RemoteBlob
from vineyard._C import RemoteBlobBuilder
from vineyard._C import _connect


class Client:
    """
    Client is responsible for managing IPC and RPC clients for Vineyard
    and provides a high-level interface to fetch objects from the Vineyard cluster.
    """

    def __init__(self, *args, **kwargs):
        """
        Connects to the vineyard IPC socket and RPC socket.

        The function supports connecting by the following priority:
        - Using an explicit IPC socket or RPC endpoint.
        - Using IPC socket or RPC endpoint from environment variables
          VINEYARD_IPC_SOCKET or VINEYARD_RPC_ENDPOINT.

        If providing an explicit IPC socket, and the IPC socket environment
        variable is also set, the explicit IPC socket will be used. Also, if
        the RPC endpoint environment variable is set at the same time, the
        RPC endpoint will be used as well.
        """
        self._ipc_client = None
        self._rpc_client = None

        try:
            client = _connect(*args, **kwargs)
        except Exception:
            client = None
        if client:
            if client.is_ipc:
                self._ipc_client = client
            else:
                self._rpc_client = client

        # Attempt to connect using environment variables
        username = kwargs.get('username', "")
        password = kwargs.get('password', "")
        session_id = kwargs.get('session_id', 0)

        if self._ipc_client is None:
            ipc_socket = os.getenv('VINEYARD_IPC_SOCKET')
            if ipc_socket and os.path.exists(ipc_socket):
                # Connect using IPC socket
                self._ipc_client = _connect(
                    ipc_socket, username=username, password=password
                )

        if self._rpc_client is None:
            rpc_endpoint = os.getenv('VINEYARD_RPC_ENDPOINT', None)
            if rpc_endpoint:
                # resolve rpc endpoint from environment variable
                hostname, port_str = rpc_endpoint.split(':')
                port = int(port_str)
                self._rpc_client = _connect(
                    hostname,
                    port,
                    session_id=session_id,
                    username=username,
                    password=password,
                )

        if self._ipc_client is None and self._rpc_client is None:
            raise ConnectionError(
                "Failed to resolve IPC socket or RPC endpoint of vineyard server from "
                "environment variables VINEYARD_IPC_SOCKET or VINEYARD_RPC_ENDPOINT."
            )

    def _get_preferred_client(self):
        if self._ipc_client:
            return self._ipc_client
        elif self._rpc_client:
            return self._rpc_client
        else:
            raise RuntimeError("No client is available.")

    # The following functions are wrappers of the corresponding functions in the
    # ClientBase class.

    def create_metadata(
        self, metadata: ObjectMeta, instance_id: int = None
    ) -> ObjectMeta:
        if instance_id is not None:
            return self._get_preferred_client().create_metadata(metadata, instance_id)
        return self._get_preferred_client().create_metadata(metadata)

    def delete(
        self,
        object: Union[ObjectID, Object, ObjectMeta, List[ObjectID]],
        force: bool = False,
        deep: bool = True,
    ) -> None:
        return self._get_preferred_client().delete(object, force, deep)

    def create_stream(self, id: ObjectID) -> None:
        return self._get_preferred_client().create_stream(id)

    def open_stream(self, id: ObjectID, mode: str) -> None:
        return self._get_preferred_client().open_stream(id, mode)

    def push_chunk(self, stream_id: ObjectID, chunk: ObjectID) -> None:
        return self._get_preferred_client().push_chunk(stream_id, chunk)

    def next_chunk_id(self, stream_id: ObjectID) -> ObjectID:
        return self._get_preferred_client().next_chunk_id(stream_id)

    def next_chunk_meta(self, stream_id: ObjectID) -> ObjectMeta:
        return self._get_preferred_client().next_chunk_meta(stream_id)

    def next_chunk(self, stream_id: ObjectID) -> Object:
        return self._get_preferred_client().next_chunk(stream_id)

    def stop_stream(self, stream_id: ObjectID, failed: bool) -> None:
        return self._get_preferred_client().stop_stream(stream_id, failed)

    def drop_stream(self, stream_id: ObjectID) -> None:
        return self._get_preferred_client().drop_stream(stream_id)

    def persist(self, object: Union[ObjectID, Object, ObjectMeta]) -> None:
        return self._get_preferred_client().persist(object)

    def exists(self, object: ObjectID) -> bool:
        return self._get_preferred_client().exists(object)

    def shallow_copy(
        self, object_id: ObjectID, extra_metadata: dict = None
    ) -> ObjectID:
        if extra_metadata:
            return self._get_preferred_client().shallow_copy(object_id, extra_metadata)
        return self._get_preferred_client().shallow_copy(object_id)

    def list_names(
        self, pattern: str, regex: bool = False, limit: int = 5
    ) -> List[str]:
        return self._get_preferred_client().list_names(pattern, regex, limit)

    def put_name(self, object: Union[Object, ObjectMeta, ObjectID], name: str) -> None:
        return self._get_preferred_client().put_name(object, name)

    def get_name(self, name: str, wait: bool = False) -> ObjectID:
        return self._get_preferred_client().get_name(name, wait)

    def drop_name(self, name: str) -> None:
        return self._get_preferred_client().drop_name(name)

    def sync_meta(self) -> None:
        return self._get_preferred_client().sync_meta()

    def migrate(self, object_id: ObjectID) -> ObjectID:
        return self._get_preferred_client().migrate(object_id)

    def clear(self) -> None:
        return self._get_preferred_client().clear()

    def memory_trim(self) -> bool:
        return self._get_preferred_client().memory_trim()

    def label(
        self,
        object_id: ObjectID,
        key_or_labels: Union[str, Dict[str, str]],
        value: str = None,
    ) -> None:
        if isinstance(key_or_labels, dict) and value is None:
            return self._get_preferred_client().label(object_id, key_or_labels)
        else:
            return self._get_preferred_client().label(object_id, key_or_labels, value)

    def evict(self, objects: List[ObjectID]) -> None:
        return self._get_preferred_client().evict(objects)

    def load(self, objects: List[ObjectID], pin: bool = False) -> None:
        return self._get_preferred_client().load(objects, pin)

    def unpin(self, objects: List[ObjectID]) -> None:
        return self._get_preferred_client().unpin(objects)

    def reset(self) -> None:
        if self._ipc_client:
            self._ipc_client.reset()
        if self._rpc_client:
            self._rpc_client.reset()

    @property
    def connected(self):
        return self._get_preferred_client().connected

    @property
    def instance_id(self):
        return self._get_preferred_client().instance_id

    @property
    def meta(self):
        return self._get_preferred_client().meta

    @property
    def status(self):
        return self._get_preferred_client().status

    def debug(self, debug: dict):
        return self._get_preferred_client().debug(debug)

    @property
    def ipc_socket(self):
        return self._get_preferred_client().ipc_socket

    @property
    def rpc_endpoint(self):
        if self._rpc_client:
            return self._rpc_client.rpc_endpoint
        return self._get_preferred_client().rpc_endpoint

    @property
    def is_ipc(self):
        return self._get_preferred_client().is_ipc

    @property
    def is_rpc(self):
        return self._get_preferred_client().is_rpc

    @property
    def version(self):
        return self._get_preferred_client().version

    # The following functions are wrappers of the corresponding functions in the
    # IPCClient and RPCClient classes.

    def create_blob(self, size: int) -> BlobBuilder:
        if self._ipc_client:
            return self._ipc_client.create_blob(size)
        warnings.warn("IPC client not available, returning None")
        return None

    def create_empty_blob(self) -> BlobBuilder:
        if self._ipc_client:
            return self._ipc_client.create_empty_blob()
        warnings.warn("IPC client not available, returning None")
        return None

    def create_remote_blob(self, blob_builder: RemoteBlobBuilder) -> ObjectID:
        if self._rpc_client:
            return self._rpc_client.create_remote_blob(blob_builder)
        warnings.warn("RPC client not available, returning None")
        return None

    def get_remote_blob(self, object_id: ObjectID, unsafe: bool = False) -> RemoteBlob:
        if self._rpc_client:
            return self._rpc_client.get_remote_blob(object_id, unsafe)
        warnings.warn("RPC client not available, returning None")
        return None

    def get_remote_blobs(
        self, object_ids: List[ObjectID], unsafe: bool = False
    ) -> List[RemoteBlob]:
        if self._rpc_client:
            return self._rpc_client.get_remote_blobs(object_ids, unsafe)
        warnings.warn("RPC client not available, returning None")
        return None

    def get_blob(self, object_id: ObjectID, unsafe: bool = False) -> Blob:
        if self._ipc_client:
            return self._ipc_client.get_blob(object_id, unsafe)
        warnings.warn("IPC client not available, returning None")
        return None

    def get_blobs(self, object_ids: List[ObjectID], unsafe: bool = False) -> List[Blob]:
        if self._ipc_client:
            return self._ipc_client.get_blobs(object_ids, unsafe)
        warnings.warn("IPC client not available, returning None")
        return None

    def get_object(self, object_id: ObjectID) -> Object:
        """
        Fetches the object associated with the given object_id from Vineyard.
        The IPC client is preferred if it's available, otherwise the RPC client
        """
        return self._fetch_object(object_id)

    def get_objects(self, object_ids: List[ObjectID]) -> List[Object]:
        objects = []
        for object_id in object_ids:
            objects.append(self.get_object(object_id))
        return objects

    def get_meta(
        self, object_id: ObjectID, sync_remote: bool = False, fetch: bool = False
    ) -> ObjectMeta:
        if self._ipc_client:
            return self._ipc_client.get_meta(object_id, sync_remote, fetch)
        elif self._rpc_client:
            return self._rpc_client.get_meta(object_id, sync_remote)
        else:
            raise RuntimeError("No IPC or RPC client available to get metadata.")

    def get_metas(
        self, object_ids: List[ObjectID], sync_remote: bool = False
    ) -> List[ObjectMeta]:
        metas = []
        for object_id in object_ids:
            metas.append(self.get_meta(object_id, sync_remote))
        return metas

    def list_objects(
        self, pattern: str, regex: bool = False, limit: int = 5
    ) -> List[ObjectID]:
        return self._get_preferred_client().list_objects(pattern, regex, limit)

    def list_metadatas(
        self, pattern: str, regex: bool = False, limit: int = 5, nobuffer: bool = False
    ) -> List[ObjectMeta]:
        return self._get_preferred_client().list_metadatas(
            pattern, regex, limit, nobuffer
        )

    def new_buffer_chunk(self, stream: ObjectID, size: int) -> memoryview:
        if self._ipc_client:
            return self._ipc_client.new_buffer_chunk(stream, size)
        warnings.warn("IPC client not available, returning None")
        return None

    def next_buffer_chunk(self, stream: ObjectID) -> memoryview:
        if self._ipc_client:
            return self._ipc_client.next_buffer_chunk(stream)
        warnings.warn("IPC client not available, returning None")
        return None

    def allocated_size(self, object_id: Union[Object, ObjectID]) -> int:
        if self._ipc_client:
            return self._ipc_client.allocated_size(object_id)
        warnings.warn("IPC client not available, returning None")
        return None

    def is_shared_memory(self, pointer: int) -> bool:
        if self._ipc_client:
            return self._ipc_client.is_shared_memory(pointer)
        warnings.warn("IPC client not available, returning None")
        return None

    def find_shared_memory(self, pointer: int) -> ObjectID:
        if self._ipc_client:
            return self._ipc_client.find_shared_memory(pointer)
        warnings.warn("IPC client not available, returning None")
        return None

    @property
    def remote_instance_id(self) -> int:
        if self._rpc_client:
            return self._rpc_client.remote_instance_id
        warnings.warn("RPC client not available, returning None")
        return None

    def close(self) -> None:
        if self._ipc_client:
            self._ipc_client.close()
        if self._rpc_client:
            self._rpc_client.close()

    def fork(self) -> 'Client':
        if self._ipc_client:
            self._ipc_client = self._ipc_client.fork()
        if self._rpc_client:
            self._rpc_client = self._rpc_client.fork()
        return self

    def _fetch_object(self, object_id: ObjectID) -> Object:
        meta = self.get_meta(object_id)

        if self._ipc_client:
            if meta.instance_id == self._ipc_client.instance_id:
                return self._ipc_client.get_object(object_id, fetch=False)
            else:
                warnings.warn(
                    f"Migrating object {object_id} from another vineyard instance "
                    f"{meta.instance_id}"
                )
                return self._ipc_client.get_object(object_id, fetch=True)
        elif self._rpc_client:
            if self._rpc_client.is_fetchable(meta):
                return self._rpc_client.get_object(object_id)
            else:
                return self._fetch_object_from_other_instance(meta)
        warnings.warn("No IPC or RPC client available, returning None")
        return None

    def _fetch_object_from_other_instance(self, meta) -> Object:
        """
        Fetches an object from another instance in the Vineyard cluster based on
        the meta information. It's used when the RPC client is not able to fetch the
        object from the current instance.
        """
        cluster_info = self._rpc_client.meta
        instance_status = cluster_info.get(meta.instance_id)

        if instance_status is None or instance_status['rpc_endpoint'] is None:
            raise RuntimeError(
                "The rpc endpoint of the vineyard instance "
                f"{meta.instance_id} is not available."
            )

        try:
            hostname, port_str = instance_status['rpc_endpoint'].split(':')
            port = int(port_str)
            new_rpc_client = _connect(hostname, port)
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to the vineyard instance at {hostname}:{port}. "
                "Make sure the vineyard instance is alive."
            ) from e

        warnings.warn(
            f"Fetching remote object {meta.id} from the remote vineyard instance "
            f"{meta.instance_id} at {hostname}:{port}."
        )
        return new_rpc_client.get_object(meta.id)


__all__ = ['Client']
