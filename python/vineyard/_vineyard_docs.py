#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from ._C import _add_doc, connect, ClientBase, IPCClient, RPCClient, \
    Object, ObjectBuilder, ObjectID, ObjectName, ObjectMeta, InstanceStatus, \
    Blob, BlobBuilder


def add_doc(target, doc):
    try:
        _add_doc(target, doc)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(target, e)


add_doc(
    connect, r'''
.. function:: connect(endpoint: str) -> IPCClient
    :noindex:

    Connect to vineyard via UNIX domain socket for IPC service:

    .. code:: python

        client = vineyard.connect('/var/run/vineyard.sock')

    Parameters:
        endpoint: str
            UNIX domain socket path to setup an IPC connection.

    Returns:
        IPCClient: The connected IPC client.

.. function:: connect(host: str, port: int or str) -> RPCClient
    :noindex:

    Connect to vineyard via TCP socket.

    Parameters:
        host: str
            Hostname to connect to.
        port: int or str
            The TCP that listened by vineyard TCP service.

    Returns:
        RPCClient: The connected RPC client.

.. function:: connect(endpoint: (str, int or str)) -> RPCClient
    :noindex:

    Connect to vineyard via TCP socket.

    Parameters:
        endpoint: tuple(str, int or str)
            Endpoint to connect to. The parameter is a tuple, in which the first element
            is the host, and the second parameter, can be int a str, is the port.

    Returns:
        RPCClient: The connected RPC client.

.. function:: connect() -> IPCClient or RPCClient
    :noindex:

    Connect to vineyard via UNIX domain socket or TCP endpoint. This method normally
    usually no arguments, and will first tries to resolve IPC socket from the
    environment variable `VINEYARD_IPC_SOCKET` and connect to it. If it fails to
    establish a connection with vineyard server, the method will tries to resolve
    RPC endpoint from the environment variable `VINEYARD_RPC_ENDPOINT`.

    If both tries are failed, this method will raise a :class:`ConnectionFailed`
    exception.

    In rare cases, user may be not sure about if the IPC socket or RPC endpoint
    is available, i.e., the variable might be :code:`None`. In such cases this method
    can accept a `None` as arguments, and do resolution as described above.

    Raises:
        ConnectionFailed
''')

add_doc(
    ObjectMeta, r'''
:class:`ObjectMeta` is the type for metadata of an :class:`Object`.
The :class:`ObjectMeta` can be treat as a *dict-like* type. If the the metadata if
the metadata obtained from vineyard, the metadata is readonly. Otherwise *key-value*
attributes or object members could be associated with the metadata to construct a
new vineyard object.

We can inspect the *key-value* attributes and members of an :class:`ObjectMeta`:

.. code:: python

    >>> meta = client.get_meta(hashmap_id)
    >>> meta
    ObjectMeta {
        "id": "0000347aebe92dd0",
        "instance_id": "0",
        ...
    }
    >>> meta['num_elements_']
    '5'
    >>> meta['entries']
    ObjectMeta {
        "id": "0000347aebe92dd0",
        "instance_id": "0",
        ...
    }

:class:`ObjectMeta` value can be iterated over:

    >>> list(k for k in meta['entries'])
    ['transient', 'num_slots_minus_one_', 'max_lookups_', 'num_elements_', 'entries_',
     'nbytes', 'typename', 'instance_id', 'id']
''')

add_doc(
    ObjectMeta.__init__, r'''
.. method:: __init__(global_: bool=False)
    :noindex:

Create an empty metadata, the metadata will be used to create a vineyard object.

Parameters
    global_: bool, if the object meta is for creating a global object.
''')

add_doc(ObjectMeta.id, r'''
The corresponding object ID of this metadata.
''')

add_doc(ObjectMeta.signature, r'''
The corresponding object signature of this metadata.
''')

add_doc(ObjectMeta.typename, r'''
The :code:`"typename"` attribute of this metadata.
''')

add_doc(ObjectMeta.nbytes, r'''
The :code:`"nbytes"` attribute of this metadata.
''')

add_doc(ObjectMeta.instance_id, r'''
The :code:`"instance_id"` of vineyard instance that the metadata been placed on.
''')

add_doc(ObjectMeta.islocal, r'''
True if the object is a local object, otherwise a global object or remote object.
''')

add_doc(ObjectMeta.isglobal, r'''
True if the object is a global object, otherwise a local object or remote object.
''')

add_doc(
    ObjectMeta.set_global, r'''
.. method: set_global(global_: bool = true)
    :noindex:

Mark the building object as a global object.

Parameters:
    global: bool, default is True
''')

add_doc(ObjectMeta.memory_usage, r'''
Get the total memory usage of buffers in this object meta.
''')

add_doc(
    ObjectMeta.__contains__, r'''
.. method: __contains__(key: str) -> bool
    :noindex:

Check if given key exists in the object metadata.

Parameters:
    key: str
        The name to be looked up.

Returns:
    bool: :code:`True` if the queried key exists in this object metadata, otherwise :code:`False`.
''')

add_doc(
    ObjectMeta.__getitem__, r'''
.. method:: __getitem__(self, key: str) -> string or Object
    :noindex:

Get meta or member's meta from metadata.

Parameters:
    key: str
        The name to be looked up.

Returns
-------
string:
    If the given key is a key of meta, returns the meta value.
Object:
    If the given key is a key of member, return the meta of this member.
''')

add_doc(
    ObjectMeta.get, r'''
.. method:: get(self, key: str, default=None) -> string or Object
    :noindex:

Get meta or member's meta from metadata, return default value if the given key is not presented.

Parameters:
    key: str
        The name to be looked up.

Returns
-------
str:
    When the given :code:`key` belongs to a metadata pair. Note that the metadata value of
    type int or float will be returned in string format as well.
ObjectMeta:
    When the given :code:`key` is mapped to a member object.

See Also:
    ObjectMeta.__getitem__
''')

add_doc(
    ObjectMeta.get_member, r'''
.. method:: get_member(self, key: str) -> Object
    :noindex:

Get member object from metadata, return None if the given key is not presented, and raise exception
RuntimeError if the given key is associated with a plain metadata, rather than member object.

Parameters:
    key: str
        The name to be looked up.

Raises:
    RuntimeError:
        When the given key is associated with a plain metadata, rather than member object.

See Also:
    ObjectMeta.__getitem__, ObjectMeta.get
''')

add_doc(
    ObjectMeta.__setitem__, r'''
.. method:: __setitem__(self, key: str, value) -> None
    :noindex:

Add a metadata to the ObjectMeta.

Parameters:
    key: str
        The name of the new metadata entry.

    value: str, int, float, bool or list of int
        The value of the new metadata entry.

        +  When the value is a :class:`str`, it will be convert to string at first.
        +  When the value is a list of str, int or float, it will be first dumpped as string
           using :code:`json.dumps`.

.. method:: __setitem__(self, key: str, ObjectID, Object or ObjectMeta) -> None
    :noindex:

Add a member object.

Parameters:
    key: str
        The name of the member object.
    object: :class:`Object`, :class:`ObjectID` or :class:`ObjectMeta`
        The reference to the member object or the object id of the member object.
''')

add_doc(
    ObjectMeta.add_member, r'''
.. method:: add_member(self, key: str, ObjectID, Object or ObjectMeta) -> None
    :noindex:

Add a member object.

Parameters:
    key: str
        The name of the member object.
    object: :class:`Object`, :class:`ObjectID` or :class:`ObjectMeta`
        The reference to the member object or the object id of the member object.
''')

add_doc(
    ObjectID, r'''
Opaque type for vineyard's object id. The object ID is generated by vineyard server, the
underlying type of :class:`ObjectID` is a 64-bit unsigned integer. Wrapper utilities
are provided to interact with the external python world.

.. code:: python

    >>> id = ObjectID("000043c5c6d5e646")
    >>> id
    000043c5c6d5e646
    >>> repr(id)
    '000043c5c6d5e646'
    >>> print(id)
    ObjectID <"000043c5c6d5e646">
    >>> int(id)
    74516723525190
''')

add_doc(
    ObjectName, r'''
Opaque type for vineyard's object name. ObjectName wraps a string, but it let users know
whether the variable represents a vineyard object, and do some smart dispatch based on that.
Wrapper utilities are provided to interact with the external python world.

.. code:: python

    >>> name = ObjectName("a_tensor")
    >>> name
    'a_tensor'
    >>> repr(name)
    "'a_tensor'"
    >>> print(name)
    a_tensor
''')

add_doc(Object, r'''
Base class for vineyard objects.
''')

add_doc(Object.id, r'''
The object id of this object.
''')

add_doc(Object.signature, r'''
The object signature of this object.
''')

add_doc(Object.meta, r'''
The metadata of this object.
''')

add_doc(Object.nbytes, r'''
The nbytes of this object.
''')

add_doc(
    Object.typename, r'''
The typename of this object. :code:`typename` is the string value of the C++ type, e.g.,
:code:`vineyard::Array<int>`, :code:`vineyard::Table`.
''')

add_doc(
    Object.member, r'''
.. method:: member(self, name: str) -> Object
    :noindex:

Get the member object of this object.

Parameters:
    name: str
        The name of the member object.

Returns:
    Object: The member object.

See Also:
    ObjectMeta.get, ObjectMeta.__getitem__
''')

add_doc(Object.islocal, r'''
Whether the object is a local object.
''')

add_doc(
    Object.ispersist, r'''
Whether the object is a persistent object. The word "persistent" means the object could
be seen by clients that connect to other vineyard server instances.
''')

add_doc(Object.isglobal, r'''
Whether the object is a global object.
''')

add_doc(ObjectBuilder, r'''
Base class for vineyard object builders.
''')

add_doc(
    ClientBase.create_metadata, r'''
.. method:: create_metadata(metadata: ObjectMeta) -> ObjectMeta
    :noindex:

Create metadata in vineyardd.

Parameters:
    metadata: ObjectMeta
        The metadata that will be created on vineyardd.

Returns:
    The result created metadata.
''')

add_doc(
    ClientBase.delete, r'''
.. method:: delete(object_id: ObjectID or List[ObjectID], force: bool = false, deep: bool = true) -> None
    :noindex:

Delete the specific vineyard object.

Parameters:
    object_id: ObjectID or list of ObjectID
        Objects that will be deleted. The :code:`object_id` can be a single :class:`ObjectID`, or a list of
        :class:`ObjectID`.
    force: bool
        Forcedly delete an object means the member will be recursively deleted even if the
        member object is also referred by others. The default value is :code:`True`.
    deep: bool
        Deeply delete an object means we will deleting the members recursively. The default
        value is :code:`True`.

        Note that when deleting objects which have *direct* blob members, the
        processing on those blobs yields a "deep" behavior.

.. method:: delete(object_meta: ObjectMeta, force: bool = false, deep: bool = true) -> None
    :noindex:

Delete the specific vineyard object.

Parameters:
    object_meta: The corresponding object meta to delete.

.. method:: delete(object: Object, force: bool = false, deep: bool = true) -> None
    :noindex:

Delete the specific vineyard object.

Parameters:
    object: The corresponding object meta to delete.
''')

add_doc(
    ClientBase.persist, r'''
.. method:: persist(object_id: ObjectID) -> None
    :noindex:

Persist the object of the given object id. After persisting, the object will be visible by clients that
connect to other vineyard server instances.

Parameters:
    object_id: ObjectID
        The object that will be persist.

.. method:: persist(object_meta: ObjectMeta) -> None
    :noindex:

Persist the given object.

Parameters:
    object_meta: ObjectMeta
        The object that will be persist.

.. method:: persist(object: Object) -> None
    :noindex:

Persist the given object.

Parameters:
    object: Object
        The object that will be persist.
''')

add_doc(
    ClientBase.exists, r'''
.. method:: exists(object_id: ObjectID) -> bool
    :noindex:

Whether the given object exists.

Parameters:
    object_id: ObjectID
        The object id to check if exists.

Returns:
    bool: :code:`True` when the specified object exists.
''')

add_doc(
    ClientBase.shallow_copy, r'''
.. method:: shallow_copy(object_id: ObjectID) -> ObjectID
    :noindex:

Create a shallow copy of the given vineyard object.

Parameters:
    object_id: ObjectID
        The vineyard object that is requested to be shallow-copied.

Returns:
    ObjectID: The object id of newly shallow-copied vineyard object.

.. method:: shallow_copy(object_id: ObjectID, extra_metadata: dict) -> ObjectID
    :noindex:

Create a shallow copy of the given vineyard object, with extra metadata.

Parameters:
    object_id: ObjectID
        The vineyard object that is requested to be shallow-copied.
    extra_metadata: dict
        Extra metadata to apply to the newly created object. The fields of extra
        metadata must be primitive types, e.g., string, number, and cannot be
        array or dict.

Returns:
    ObjectID: The object id of newly shallow-copied vineyard object.
''')

add_doc(
    ClientBase.put_name, r'''
.. method:: put_name(object: ObjectID or ObjectMeta or Object,
                     name: str or ObjectName) -> None
    :noindex:

Associate the given object id with a name. An :class:`ObjectID` can be associated with more
than one names.

Parameters:
    object_id: ObjectID
    name: str
''')

add_doc(
    ClientBase.get_name, r'''
.. method:: get_name(name: str or ObjectName, wait: bool = False) -> ObjectID
    :noindex:

Get the associated object id of the given name.

Parameters:
    name: str
        The name that will be queried.
    wait: bool
        Whether to wait util the name appears, if wait, the request will be blocked
        until the name been registered.

Return:
    ObjectID: The associated object id with the name.
''')

add_doc(
    ClientBase.drop_name, r'''
.. method:: drop_name(name: str or ObjectName) -> None
    :noindex:

Remove the association of the given name.

Parameters:
    name: str
        The name that will be removed.
''')

add_doc(ClientBase.sync_meta, r'''
.. method:: sync_meta() -> None
    :noindex:

Synchronize remote metadata to local immediately.
''')

add_doc(ClientBase.connected, r'''
Whether the client instance has been connected to the vineyard server.
''')

add_doc(ClientBase.instance_id, r'''
The instance id of the connected vineyard server.
''')

add_doc(
    ClientBase.meta, r'''
The metadata information of the vineyard server. The value is a  nested dict, the first-level
key is the instance id, and the second-level key is the cluster metadata fields.

.. code:: python

    >>> client.meta
    {
        14: {
            'hostid': '54058007061210',
            'hostname': '127.0.0.1',
            'timestamp': '6882550126788354072'
        },
        15: {
            'hostid': '48843417291806',
            'hostname': '127.0.0.1',
            'timestamp': '6882568290204737414'
        }
    }
''')

add_doc(
    ClientBase.status, r'''
The status the of connected vineyard server, returns a :class:`InstanceStatus`.

See Also:
    InstanceStatus
''')

add_doc(ClientBase.ipc_socket, r'''
The UNIX domain socket location of connected vineyard server.
''')

add_doc(ClientBase.rpc_endpoint, r'''
The RPC endpoint of the connected vineyard server.
''')

add_doc(ClientBase.version, r'''
The version number string of connected vineyard server, in the format of semver: MAJOR.MINOR.PATCH.
''')

add_doc(IPCClient, r'''
IPC client that connects to vineyard instance's UNIX domain socket.
''')

add_doc(
    IPCClient.create_blob, r'''
.. method:: create_blob(size: int) -> Blob
    :noindex:

Allocate a blob in vineyard server.

Parameters:
    size: int
        The size of blob that will be allocated on vineyardd.

Returns:
    BlobBuilder
''')

add_doc(
    IPCClient.create_empty_blob, r'''
.. method:: create_empty_blob() -> Blob
    :noindex:

Create an empty blob in vineyard server.

Returns:
    Blob
''')

add_doc(
    IPCClient.get_object, r'''
.. method:: get_object(object_id: ObjectID) -> Object
    :noindex:

Get object from vineyard.

Parameters:
    object_id: ObjectID
        The object id to get.

Returns:
    Object
''')

add_doc(
    IPCClient.get_objects, r'''
.. method:: get_objects(object_ids: List[ObjectID]) -> List[Object]
    :noindex:

Get multiple objects from vineyard.

Paramters:
    object_ids: List[ObjectID]

Returns:
    List[Object]
''')

add_doc(
    IPCClient.get_meta, r'''
.. method:: get_meta(object_id: ObjectID, sync_remote: bool = False) -> ObjectMeta
    :noindex:

Get object metadata from vineyard.

Parameters:
    object_id: ObjectID
        The object id to get.
    sync_remote: bool
        If the target object is a remote object, :code:`code_remote=True` will force
        a meta synchronization on the vineyard server. Default is :code:`False`.

Returns:
    ObjectMeta
''')

add_doc(
    IPCClient.get_metas, r'''
.. method:: get_metas(object_ids: List[ObjectID], sync_remote: bool = False) -> List[ObjectMeta]
    :noindex:

Get metadatas of multiple objects from vineyard.

Paramters:
    object_ids: List[ObjectID]
        The object ids to get.
    sync_remote: bool
        If the target object is a remote object, :code:`code_remote=True` will force
        a meta synchronization on the vineyard server. Default is :code:`False`.

Returns:
    List[ObjectMeta]
''')

add_doc(
    IPCClient.list_objects, r'''
.. method:: list_objects(pattern: str, regex: bool = False, limit: int = 5) -> List[Object]
    :noindex:

List all objects in current vineyard server.

Parameters:
    pattern: str
        The pattern string that will be matched against the object's typename.
    regex: bool
        Whether the pattern is a regex expression, otherwise the pattern will be used as
        wildcard pattern. Default value is False.
    limit: int
        The limit to list. Default value is 5.

Returns:
    List[Object]
''')

add_doc(
    IPCClient.list_metadatas, r'''
.. method:: list_metadatas(pattern: str, regex: bool = False, limit: int = 5, nobuffer: bool = False) -> List[Object]
    :noindex:

List all objects in current vineyard server.

Parameters:
    pattern: str
        The pattern string that will be matched against the object's typename.
    regex: bool
        Whether the pattern is a regex expression, otherwise the pattern will be used as
        wildcard pattern. Default value is False.
    limit: int
        The limit to list. Default value is 5.
    nobuffer: bool
        Whether to fill the buffers in returned object metadatas. Default value is False.

Returns:
    List[Object]
''')

add_doc(
    IPCClient.allocated_size, r'''
.. method:: allocated_size(target: Object or ObjectID) -> int
    :noindex:

Get the allocated size of the given object.

Parameters:
    target: Object or ObjectID
        The given Object.

Returns:
    int
''')

add_doc(
    IPCClient.is_shared_memory, r'''
.. method:: allocated_size(target: ptr) -> bool
    :noindex:

Check if the address is on the shared memory region.

Parameters:
    target: address, in int format
        The given address.

Returns:
    bool
''')

add_doc(IPCClient.close, r'''
Close the client.
''')

add_doc(
    RPCClient, r'''
RPC client that connects to vineyard instance's RPC endpoints.

The RPC client can only access the metadata of objects, any access to the blob payload
will trigger a :code:`RuntimeError` exception.
''')

add_doc(
    RPCClient.get_object, r'''
.. method:: get_object(object_id: ObjectID) -> Object
    :noindex:

Get object from vineyard.

Parameters:
    object_id: ObjectID
        The object id to get.

Returns:
    Object
''')

add_doc(
    RPCClient.get_objects, r'''
.. method:: get_objects(object_ids: List[ObjectID]) -> List[Object]
    :noindex:

Get multiple objects from vineyard.

Paramters:
    object_ids: List[ObjectID]

Returns:
    List[Object]
''')

add_doc(
    RPCClient.get_meta, r'''
.. method:: get_meta(object_id: ObjectID) -> ObjectMeta
    :noindex:

Get object metadata from vineyard.

Parameters:
    object_id: ObjectID
        The object id to get.

Returns:
    ObjectMeta
''')

add_doc(
    RPCClient.get_metas, r'''
.. method:: get_metas(object_ids: List[ObjectID] -> List[ObjectMeta]
    :noindex:

Get metadatas of multiple objects from vineyard.

Paramters:
    object_ids: List[ObjectID]

Returns:
    List[ObjectMeta]
''')

add_doc(
    RPCClient.list_objects, r'''
.. method:: list_objects(pattern: str, regex: bool = False, limit: int = 5) -> List[Object]
    :noindex:

List all objects in current vineyard server.

Parameters:
    pattern: str
        The pattern string that will be matched against the object's typename.
    regex: bool
        Whether the pattern is a regex expression, otherwise the pattern will be used as
        wildcard pattern. Default value is False.
    limit: int
        The limit to list. Default value is 5.

Returns:
    List[Object]
''')

add_doc(
    RPCClient.list_metadatas, r'''
.. method:: list_metadatas(pattern: str, regex: bool = False, limit: int = 5, nobuffer: bool = False) -> List[Object]
    :noindex:

List all objects in current vineyard server.

Parameters:
    pattern: str
        The pattern string that will be matched against the object's typename.
    regex: bool
        Whether the pattern is a regex expression, otherwise the pattern will be used as
        wildcard pattern. Default value is False.
    limit: int
        The limit to list. Default value is 5.
    nobuffer: bool
        Whether to fill the buffers in returned object metadatas. Default value is False.

Returns:
    List[Object]
''')

add_doc(RPCClient.close, r'''
Close the client.
''')

add_doc(RPCClient.remote_instance_id, r'''
The instance id of the connected remote vineyard server.
''')

add_doc(
    InstanceStatus, r'''
:class:`InstanceStatus` represents the status of connected vineyard instance, including
the instance identity, memory statistics and workloads on this instance.

.. code:: python

    >>> status = client.status
    >>> print(status)
    InstanceStatus:
        instance_id: 5
        deployment: local
        memory_usage: 360
        memory_limit: 268435456
        deferred_requests: 0
        ipc_connections: 1
        rpc_connections: 0
    >>> status.instance_id
    5
    >>> status.deployment
    'local'
    >>> status.memory_usage
    360
    >>> status.memory_limit
    268435456
    >>> status.deferred_requests
    0
    >>> status.ipc_connections
    1
    >>> status.rpc_connections
    0
''')

add_doc(InstanceStatus.instance_id, r'''
Return the instance id of vineyardd that the client is connected to.
''')

add_doc(
    InstanceStatus.deployment, r'''
The deployment mode of the connected vineyardd cluster, can be :code:`"local"` and
:code:`"distributed"`.
''')

add_doc(InstanceStatus.memory_usage, r'''
Report memory usage (in bytes) of current vineyardd instance.
''')

add_doc(InstanceStatus.memory_limit, r'''
Report memory limit (in bytes) of current vineyardd instance.
''')

add_doc(InstanceStatus.deferred_requests, r'''
Report number of waiting requests of current vineyardd instance.
''')

add_doc(InstanceStatus.ipc_connections, r'''
Report number of alive IPC connections on the current vineyardd instance.
''')

add_doc(InstanceStatus.rpc_connections, r'''
Report number of alive RPC connections on the current vineyardd instance.
''')

add_doc(Blob, r'''
:class:`Blob` in vineyard is a consecutive readonly shared memory.
''')

add_doc(Blob.size, r'''
Size of the blob.
''')

add_doc(Blob.empty, r'''
Whether the blob is an empty blob, i.e., the size of this blob is 0.
''')

add_doc(Blob.__len__, r'''
The size of this blob.
''')

add_doc(Blob.address, r'''
The memory address value of this blob.
''')

add_doc(Blob.buffer, r'''
The readonly buffer hebind this blob. The result buffer has type :code:`pyarrow::Buffer`.
''')

add_doc(
    BlobBuilder, r'''
:class:`BlobBuilder` is the builder for creating a finally immutable blob in vineyard server.

A :class:`BlobBuilder` can only be explicitly created using the :meth:`IPCClient.create_blob`.

See Also:
    IPCClient.create_blob
    IPCClient.create_empty_blob
''')

add_doc(BlobBuilder.id, r'''
ObjectID of this blob builder.
''')

add_doc(BlobBuilder.size, r'''
Size of this blob builder.
''')

add_doc(BlobBuilder.abort, r'''
Abort the blob builder if it is not sealed yet.
''')

add_doc(
    BlobBuilder.copy, r'''
.. method:: copy(self, offset: int, ptr: int, size: int)
    :noindex:

Copy the given address to the given offset.
''')

add_doc(BlobBuilder.address, r'''
The memory address value of this blob builder.
''')

add_doc(
    BlobBuilder.buffer, r'''
The writeable buffer hebind this blob builder. The result buffer has type :code:`pyarrow::Buffer`,
and it is a mutable one.
''')
