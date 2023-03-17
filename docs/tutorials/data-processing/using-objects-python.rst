.. _using-objects-python:

Sharing Python objects with vineyard
------------------------------------

As we mentioned in :ref:`vineyard-objects`, for each object in vineyard, it
consists of two parts:

1. The data payload stored in the corresponding vineyard instance locally
2. The hierarchical meta data shared across the vineyard cluster

In particular, ``Blob`` is the unit where the data payload lives in a vineyard
instance.
A blob object holds a segment of memory in the bulk store of the vineyard
instance, so that users can save their local buffer into a blob and
get the blob later in another process in a zero-copy fashion through
memory mapping.

.. code:: python

    >>> payload = b"Hello, World!"
    >>> blob_id = client.put(payload)
    >>> blob = client.get_object(blob_id)
    >>> print(blob.typename, blob.size, blob)

.. code:: console

    vineyard::Blob 28 Object <"o800000011cfa7040": vineyard::Blob>

On the other hand, the hierarchical meta data of vineyard objects are
shared across the cluster. In the following example, for simplicity,
we launch a vineyard cluster with
two vineyard instances in the same machine, although in practice,
these vineyard instances are launched distributively on each machine of the cluster.

.. code:: console

    $ python3 -m vineyard --socket /var/run/vineyard.sock1
    $ python3 -m vineyard --socket /var/run/vineyard.sock2

Then we can create a distributed pair of arrays in vineyard with the
first array stored in the first vineyard instance which listens to ipc_socket
``/var/run/vineyard.sock1``, and the second array stored in the second instance
listening to ipc_socket ``/var/run/vineyard.sock2``.

.. code:: python

    >>> import numpy as np
    >>> import vineyard
    >>> import vineyard.data.tensor

    >>> # build the first array in the first vineyard instance
    >>> client1 = vineyard.connect('/var/run/vineyard.sock1')
    >>> id1 = client1.put(np.zeros(8))
    >>> # persist the object to make it visible to form the global object
    >>> client1.persist(id1)

    >>> # build the second array in the second vineyard instance
    >>> client2 = vineyard.connect('/var/run/vineyard.sock2')
    >>> id2 = client2.put(np.ones(4))
    >>> # persist the object to make it visible to form the global object
    >>> client2.persist(id2)

    >>> # build the pair from client1
    >>> obj1 = client1.get_object(id1)
    >>> obj2 = client2.get_object(id2)
    >>> id_pair = client1.put((obj1, obj2))

    >>> # get the pair object from client2
    >>> obj_pair = client2.get_object(id_pair)
    >>> print(obj_pair.first.typename, obj_pair.first.size(), obj_pair.second.size())

.. code:: console

    vineyard::Array 8 4

.. code:: console

    >>> # get the pair value from client2
    >>> value_pair = client2.get(id_pair)
    >>> print(value_pair)

.. code:: console

    (None, [1, 1, 1, 1])

Here we can get the meta data of the pair object from ``client2``
though ``client1`` created it, but we can't get the payload of the
first element of the pair from ``client2``, since it is stored locally
in the first vineyard instance.
