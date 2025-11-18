.. _using-objects-python:

Sharing Python Objects with Vineyard
------------------------------------

As discussed in :ref:`vineyard-objects`, each object in Vineyard consists of two parts:

1. The data payload, which is stored locally in the corresponding Vineyard instance
2. The hierarchical metadata, which is shared across the entire Vineyard cluster

Specifically, a ``Blob`` represents the unit where the data payload resides within a
Vineyard instance. A blob object holds a segment of memory in the bulk store of the
Vineyard instance, allowing users to save their local buffer into a blob and later
retrieve the blob in another process using a zero-copy approach through memory mapping.

.. code:: python

    >>> payload = b"Hello, World!"
    >>> blob_id = client.put(payload)
    >>> blob = client.get_object(blob_id)
    >>> print(blob.typename, blob.size, blob)

.. code:: console

    vineyard::Blob 28 Object <"o800000011cfa7040": vineyard::Blob>

On the other hand, the hierarchical metadata of Vineyard objects is shared across
the entire cluster. In the following example, for the sake of simplicity, we
launch a Vineyard cluster consisting of two Vineyard instances on the same machine.
However, in real-world scenarios, these Vineyard instances would be distributed
across multiple machines within the cluster.

.. code:: console

    $ python3 -m vineyard --socket /var/run/vineyard.sock1
    $ python3 -m vineyard --socket /var/run/vineyard.sock2

With this setup, we can create a distributed pair of arrays in Vineyard, where
the first array is stored in the first Vineyard instance listening to the IPC socket
``/var/run/vineyard.sock1``, and the second array is stored in the second instance
listening to the IPC socket ``/var/run/vineyard.sock2``.

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

In this example, we can access the metadata of the pair object from ``client2``
even though it was created by ``client1``. However, we cannot retrieve the payload
of the first element of the pair from ``client2`` because it is stored locally
in the first Vineyard instance.
