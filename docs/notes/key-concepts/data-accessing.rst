Data Accessing
==============

Vineyard is designed to support distributed object sharing and offers both IPCClient
and RPCClient for efficient data access. This section will guide you through various
methods of accessing objects within vineyard. For more information on vineyard object
basics, please refer to :ref:`metadata-and-payloads` and :ref:`distributed-objects`.

IPCClient vs. RPCClient
-----------------------

As depicted in the above figure, data is partitioned across different vineyard
instances. The concept of zero-copy sharing was explained in :ref:`architecture-of-vineyard`.
Memory mapping is only available for clients on the same instance, while metadata
is globally synchronized and accessible from clients connected to instances on other hosts.

Vineyard provides two clients to support IPC and RPC scenarios:

- IPC Client

  - Can only connect to instances deployed on the same host.
  - Offers full support for local data access. Accessing local blobs is enabled
    by zero-copy memory mapping.

- RPC Client

  - Can connect to any instance with an enabled RPC endpoint.
  - Provides limited support for remote data access. Creating and fetching remote
    blobs incurs considerable network transfer overhead.

Local vs. Remote
^^^^^^^^^^^^^^^^

Distributed shared objects are typically partitioned, with each vineyard instance managing
some chunks of the entire object. As shown in :ref:`distributed-objects`, a :code:`GlobalTensor`
is partitioned into three chunks, and each instance holds one chunk of type :code:`Tensor`.

**From the perspective of computing engines**, distributed computing engines launch
workers on vineyard instances. Each worker connects to the co-located local instance and
is responsible for processing chunks in that local instance. For example, when starting a Dask
cluster on a vineyard cluster as illustrated in the picture above, each Dask worker is responsible
for executing computations on its local chunks. Some computing tasks require communication between
workers, such as aggregation. In these cases, the communication is performed by the computing
engine itself (in this case, the Dask cluster).

.. tip::

    We assume that the computing engines built upon vineyard are responsible for scheduling
    tasks based on their awareness of the underlying data partitioning within the vineyard
    cluster.

    This design is well-suited for commonly-used modern computing engines,such as GraphScope,
    Spark, Presto, Dask, Mars, and Ray.

Local Objects
-------------

Creating and accessing local objects in vineyard can be easily achieved using :code:`put` and :code:`get` methods (see
:meth:`vineyard.IPCClient.put` and :meth:`vineyard.IPCClient.get`).

.. code:: python
   :caption: Effortlessly create and access local objects using :code:`put` and :code:`get`

    >>> import pandas as pd
    >>> import vineyard
    >>> import numpy as np
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
              0         1
    0  0.534487  0.261941
    1  0.901056  0.441583
    2  0.687568  0.671564
    ...

Vineyard provides low-level APIs to operate on metadatas and raw blobs as well.

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

Using blobs
^^^^^^^^^^^

Vineyard offers low-level APIs for creating and accessing local blobs with enhanced efficiency:

- :meth:`vineyard.IPCClient.create_blob`: creates an empty :class:`vineyard.BlobBuilder` for
  you to fill with data.
- :meth:`vineyard.IPCClient.get_blob`: retrieves a :class:`vineyard.Blob` from the vineyard
  cluster using zero-copy techniques.
- :meth:`vineyard.IPCClient.get_blobs`: fetches a list of :code:`List[vineyard.Blob]` from the
  vineyard cluster, also utilizing zero-copy methods.

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

Creating and accessing remote objects in vineyard can be easily achieved using :code:`put` and :code:`get` methods (see
:meth:`vineyard.RPCClient.put` and :meth:`vineyard.RPCClient.get`).

.. code:: python
   :caption: Effortlessly create and access remote objects using :code:`put` and :code:`get`

    >>> import pandas as pd
    >>> import vineyard
    >>> import numpy as np
    >>>
    >>> vineyard_rpc_client = vineyard.connect("localhost", 9600)
    >>>
    >>> df = pd.DataFrame(np.random.rand(10, 2))
    >>>
    >>> # put object into vineyard
    >>> r = vineyard_rpc_client.put(df)
    >>> r, type(r)
    (o000a45730a85f8fe, vineyard._C.ObjectID)
    >>>
    >>> # get object from vineyard using object id
    >>> data = vineyard_rpc_client.get(r)
    >>> data
              0         1
    0  0.884227  0.576031
    1  0.863040  0.069815
    2  0.297906  0.911874
    ...

The RPC client enables inspection of remote object metadata and facilitates operations on blobs
within the remote cluster, while taking into account the associated network transfer costs.

Inspecting metadata
^^^^^^^^^^^^^^^^^^^

The method :meth:`vineyard.RPCClient.get_meta` allows you to access object metadata in a similar
manner to :meth:`vineyard.IPCClient.get_meta`, but with the added capability of connecting to a
remote instance.

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

Using remote blobs
^^^^^^^^^^^^^^^^^^

However, due to the absence of memory sharing between hosts, zero-copy data sharing is not feasible when
connecting to a vineyard instance that is not deployed on the same host as the client. Transferring data
over the network incurs significant costs, and vineyard requires users to explicitly issue a :code:`migrate`
command to move data from the remote instance to the local instance. For more details, please refer to
:ref:`Object Migration in Vineyard <client-side>`.

For added convenience, we also provide APIs to fetch remote blobs to the local client by transferring
payloads over the network.

- :meth:`vineyard.RPCClient.create_remote_blob`: put a **filled** remote blob builder
  :class:`vineyard.RemoteBlobBuilder` to connected remote instance.
- :meth:`vineyard.RPCClient.get_remote_blob`: obtain a remote blob :class:`vineyard.RemoteBlob`
  from the vineyard cluster by copying over the network.
- :meth:`vineyard.RPCClient.get_remote_blobs`: obtain a set of remote blobs
  :code:`List[vineyard.RemoteBlob]` from the vineyard cluster by copying over the network.

.. warning::

    Note that the :code:`remote` in the above APIs means the blob will be transferred using
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
    >>> remote_blob_meta = vineyard_rpc_client.create_remote_blob(remote_buffer_builder)

.. code:: python
   :caption: Accessing remote blobs

    >>> # get the remote blob from vineyard using object id
    >>> remote_blob = vineyard_rpc_client.get_remote_blob(remote_blob_meta.id)
    >>> remote_blob, type(remote_blob)
    (<vineyard._C.RemoteBlob at 0x142204870>, vineyard._C.RemoteBlob)
    >>>
    >>> # inspect the value of remote blob
    >>> bytes(memoryview(remote_blob))
    b'abcdefgh1234567890uvwxyz'

.. warning::

    The APIs for creating blobs in :class:`vineyard.IPCClient` and :class:`vineyard.RPCClient`
    have subtle differences. The :meth:`vineyard.IPCClient.create_blob` method first allocates
    a shared memory buffer to create an empty blob builder, allowing the user to fill the buffer
    and then seal it. In contrast, the :meth:`vineyard.RPCClient.create_remote_blob` method
    creates a remote blob builder on-the-fly, enabling the user to fill the buffer and subsequently
    use the client API to send the :code:`remote_buffer_builder` to the remote instance.

Utilizing Distributed Objects
-----------------------------

In the illustration at the beginning of this section, we demonstrate that vineyard is capable of sharing
distributed objects partitioned across multiple hosts. Accessing these distributed objects
in vineyard can be achieved through two distinct approaches:

- Inspecting metadata using the :code:`RPCClient`:

  The metadata of global objects can be examined using the :class:`vineyard.RPCClient`. This allows
  computing engines to understand the distribution of partitions of global tensors using the
  RPCClient, and subsequently schedule jobs over those chunks based on the distribution information.

  Mars employs this method to consume distributed tensors and dataframes in vineyard.

- Accessing local partitions of global objects using the :code:`IPCClient`:

  Another prevalent pattern for accessing shared global objects involves launching a worker on each
  instance where the global object is partitioned. Then, using the :class:`vineyard.IPCClient`,
  workers can obtain the local partitions of the global object. Each worker is responsible for
  processing its local partitions.

  This pattern is commonly utilized in many computing engines that have been integrated with
  vineyard, such as GraphScope and Presto.
