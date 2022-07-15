Shared Data Accessing
=====================

Vineyard supports distributed object sharing by-design, and provides both the IPCClient
and RPCClient for data accessing. You would learn how accessing objects inside vineyard in
various ways. For vineyard objects basics, please refer to :ref:`metadata-and-payloads`.

.. figure:: ../images/vineyard_deployment.jpg
   :alt: Data Partitioning in Vineyard
   :width: 60%

   Data Partitioning in Vineyard

**The distributed shared objects are generally partitioned** and each vineyard instance manages
some chunks of the whole object. As shown in the picture above, a :code:`GlobalTensor` is
partitioned into three chunks and each instance hold one chunk of type :code:`Tensor`.

**From the perspective of computing engines**, the distributed computing engines launches
workers upon the vineyard instances. Each worker connects the co-located local instance and
is responsible for processing chunks in the local instance. E.g., we start a Dask cluster on
vineyard cluster illustrated in the picture above, and each Dask worker is responsible for
executing computation on its local chunks. Some computing tasks require communication between
workers, e.g., aggregation. In such cases the communication is performed by the computing
engines itself (here the Dask cluster).

.. tip::

    We assume the computing engines upon vineyard is responsible to schedule the tasks based
    on the awareness of the underlying data partitioning inside the vineyard cluster.

    Such a design fits commonly-used modern computing engines, e.g., GraphScope, Spark, Presto,
    Dask, Mars and Ray pretty well.

IPCClient vs. RPCClient
-----------------------

From the above figure, we can see that the data is partitioned across different vineyard
instances. We have illustrated idea behind zero-copy sharing in :ref:`architecture-of-vineyard`.
Memory mapping is only available from the clients on the same instance whereas the metadata
is globally synchronized and available from clients that connect to instances on other hosts.

Vineyard provides two clients to support the IPC and RPC scenarios:

- IPC Client

  - Can only be connected to instances that deployed on the same hosts.
  - Full support for local data accessing. Accessing local blobs can be done in a zero-copy
    enabled by memory mapping.

- RPC Client

  - Can be connected to any instance whose RPC endpoint is enabled
  - Limited support for remote data accessing. Creating and fetching remote blobs yields a
    considerable network transferring overhead.

Local Objects
-------------

Creating and accessing local objects in vineyard is easy as :code:`put` and :code:`get` (see
:meth:`vineyard.IPCClient.put` and :meth:`vineyard.IPCClient.get`).

.. code:: python
   :caption: Creating and accessing local objects is easy as :code:`put` and :code:`get` 

    >>> import pandas as pd
    >>> import vineyard
    >>>
    >>> vineyard_ipc_client = vineyard.connect("/tmp/vineyard.sock")
    >>>
    >>> df = pd.DataFrame(np.random.rand(10, 2))
    >>>
    >>> # put object into vineyard
    >>> r = vineyard_ipc_client.put(df)
    >>> r, type(r)
    (o00053008257020f8, vineyard._C.ObjectID)
    >>>
    >>> # get object from vineyard using object id
    >>> data = vineyard_ipc_client.get(r)
    >>> data
    In [10]: data
    Out[10]:
              0         1
    0  0.534487  0.261941
    1  0.901056  0.441583
    2  0.687568  0.671564
    ...

Vineyard provides low level APIs to operate on metadatas and raw blobs as well.

Accessing metadatas
^^^^^^^^^^^^^^^^^^^

The method :meth:`vineyard.IPCClient.get_meta` can be used to inspect metadata in the
vineyard cluster, which returns a :class:`vineyard.ObjectMeta` value:

.. code:: python
   :caption: Accessing metadata in vineyard 

    >>> meta = vineyard_ipc_client.get_meta(r)
    >>> meta.id
    o00053008257020f8
    >>> meta.instance_id
    0
    >>> meta.typename
    'vineyard::DataFrame'
    >>> meta
    {
        "instance_id": 0,
        "nbytes": 0,
        "signature": 1460186430481176,
        "transient": true,
        "typename": "vineyard::DataFrame"
        "__values_-value-0": {
            "global": false,
            "id": "o0005300822f54d1c",
            "instance_id": 0,
            "nbytes": 80,
            "order_": "\"F\"",
            "shape_": "[10]",
            "signature": 1460186388165810,
            "transient": true,
            "typename": "vineyard::Tensor<double>",
            "value_type_": "float64",
            "value_type_meta_": "<f8"
            "buffer_": {
                "id": "o8005300822d858df",
                "typename": "vineyard::Blob"
                ...

Creating and accessing blobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vineyard also provides low level APIs to create and access local blobs,

- :meth:`vineyard.IPCClient.create_blob`: create a empty blob builder :class:`vineyard.BlobBuilder`
  and **then fill it**.
- :meth:`vineyard.IPCClient.get_blob`: obtain a blob :class:`vineyard.Blob` from the vineyard
  cluster in zero copy fashion.
- :meth:`vineyard.IPCClient.get_blobs`: obtain a set of blobs :code:`List[vineyard.Blob]` from
  the vineyard cluster in zero copy fashion.

.. code:: python
   :caption: Creating local blobs

    >>> import vineyard
    >>> vineyard_ipc_client = vineyard.connect("/tmp/vineyard.sock")
    >>>
    >>> # mock a data
    >>> payload = b'abcdefgh1234567890uvwxyz'
    >>>
    >>> # create a blob builder
    >>> buffer_builder = vineyard_ipc_client.create_blob(len(payload))
    >>>
    >>> # copy the mocked data into the builder
    >>> buffer_builder.copy(0, payload)
    >>>
    >>> # seal the builder then we will get a blob
    >>> blob = buffer_builder.seal(vineyard_ipc_client)

.. code:: python
   :caption: Accessing local blobs

    >>> # get the blob from vineyard using object id
    >>> blob = vineyard_ipc_client.get_blob(blob.id)
    >>> blob, type(blob)
    (Object <"o800532e4ab1f2087": vineyard::Blob>, vineyard._C.Blob)
    >>>
    >>> # inspect the value
    >>> bytes(memoryview(blob))
    b'abcdefgh1234567890uvwxyz'

Remote Objects
--------------

The RPC client can be used to inspect the remote object metadata and operate blobs on remote
cluster with network transferring cost.

Accessing object metadata using RPCClient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The method :meth:`vineyard.RPCClient.get_meta` can be used to access the object metadata,
like :meth:`vineyard.IPCClient.get_meta`, but could be used over the connection to a remote
instance,

.. code:: python
   :caption: Metadata accessing using RPCClient

    >>> import vineyard
    >>> vineyard_rpc_client = vineyard.connect("localhost", 9600)
    >>>
    >>> # the `r` from the above "Local Objects" section 
    >>> meta = vineyard_rpc_client.get_meta(r)
    >>> meta.id
    o00053008257020f8
    >>> meta.instance_id
    0
    >>> meta.typename
    'vineyard::DataFrame'

Operating blobs using RPCClient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, as lacking of memory sharing between hosts, the zero-copy data sharing is not
possible when connecting to a vineyard instance that isn't deployed on the same host with
the client. Moving data over network yields considerable cost and vineyard requests the
user to issue a :code:`migrate` command explicitly to move the data from the remote
instance to the local instance, see also :ref:`Object Migration in Vineyard <client-side>`.

For convenience, we also provides APIs to fetch remote blobs to local client by transferring
the payloads over network,

- :meth:`vineyard.RPCClient.create_remote_blob`: put a **filled** remote blob builder
  :class:`vineyard.RemoteBlobBuilder` to connected remote instance.
- :meth:`vineyard.RPCClient.get_remote_blob`: obtain a remote blob :class:`vineyard.RemoteBlob`
  from the vineyard cluster by copying over the network.
- :meth:`vineyard.RPCClient.get_remote_blobs`: obtain a set of remote blobs
  :code:`List[vineyard.RemoteBlob]` from the vineyard cluster by copying over the network.

.. warning::

    Note that the :code:`remote` in above APIs means the blob will be transferred using
    TCP network. For large blobs, it implies a significant cost of time.

.. code:: python
   :caption: Creating remote blobs

    >>> import vineyard
    >>> vineyard_rpc_client = vineyard.connect("localhost", 9600)
    >>>
    >>> # mock a data
    >>> payload = b'abcdefgh1234567890uvwxyz'
    >>>
    >>> # create an empty blob builder
    >>> remote_buffer_builder = vineyard.RemoteBlobBuilder(len(payload))
    >>>
    >>> # copy the mocked data into the builder
    >>> remote_buffer_builder.copy(0, payload)
    >>>
    >>> # create the remote blob using the RPCClient, with the `remote_buffer_builder` as argument
    >>> remote_blob_id = vineyard_rpc_client.create_remote_blob(remote_buffer_builder)

.. code:: python
   :caption: Accessing remote blobs

    >>> # get the remote blob from vineyard using object id
    >>> remote_blob = vineyard_rpc_client.get_remote_blob(remote_blob_id)
    >>> remote_blob, type(remote_blob)
    (<vineyard._C.RemoteBlob at 0x142204870>, vineyard._C.RemoteBlob)
    >>>
    >>> # inspect the value of remote blob
    >>> bytes(memoryview(remote_blob))
    b'abcdefgh1234567890uvwxyz'

.. warning::

    The blob creation API on the :class:`vineyard.IPCClient` and :class:`vineyard.RPCClient`
    differs slightly. The :meth:`vineyard.IPCClient.create_blob` creates a empty blob builder
    by allocating a shared memory buffer first, then let the user to fill the buffer, and
    finally seal the buffer. However the :meth:`vineyard.RPCClient.create_remote_blob` creates
    a remote blob builder on-the-fly first, then let the user to fill the buffer, and finally
    using the client API to send to buffer (the :code:`remote_buffer_builder`) to the remote
    instance.

Distributed Objects
-------------------

In the picture at the beginning of this section, we show that vineyard is capable to share
distributed objects that partitioned across multiple hosts. Accessing the distributed objects
in vineyard involves the following two different ways:

- Accessing the metadata using the :code:`RPCClient`:

  The metadata of global objects can be inspected using the :class:`vineyard.RPCClient`, i.e.,
  the computing engines can know the distribution of partitions of global tensor using the
  RPCClient, then schedule jobs over those chunks by respecting the distribution information.

  Mars works in such a way to consume distributed tensors and dataframes in vineyard.

- Accessing the local partitions of global objects using the :code:`IPCClient`:

  Another common pattern of accessing shared global objects is launching a worker on each
  instance where the global object is partitioned, and then using the :class:`vineyard.IPCClient`
  to get the local partitions of the global object. Each worker is responsible to process
  its local partitions.

  Such a pattern is commonly used in many computing engines that has been integrated with
  vineyard, e.g., GraphScope and Presto.

Accessing Streams
-----------------

Stream is an abstraction that designed to help the pipelining between two consecutive
big-data analytical tasks.

For details about accessing streams in vineyard, please refer to :ref:`streams-in-vineyard`.
