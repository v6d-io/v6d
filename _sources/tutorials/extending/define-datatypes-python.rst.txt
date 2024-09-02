.. _define-python-types:

Define Data Types in Python
---------------------------

Objects
^^^^^^^

As discussed in :ref:`vineyard-objects`, each object in vineyard comprises two components:

1. The data payload, which is stored locally within the corresponding vineyard instance
2. The hierarchical meta data, which is shared across the entire vineyard cluster

Specifically, a ``Blob`` represents the unit where the data payload resides in a vineyard
instance. A blob object contains a segment of memory in the bulk store of the vineyard
instance, allowing users to save their local buffer into a blob and later retrieve the
blob in another process using a zero-copy approach through memory mapping.

.. code:: python

    >>> payload = b"Hello, World!"
    >>> blob_id = client.put(payload)
    >>> blob = client.get_object(blob_id)
    >>> print(blob.typename, blob.size, blob)

.. code:: console

    vineyard::Blob 28 Object <"o800000011cfa7040": vineyard::Blob>

On the other hand, vineyard objects' hierarchical meta data is shared across the entire
cluster. In the following example, for the sake of simplicity, we will launch a vineyard
cluster with two vineyard instances on the same machine. However, in real-world scenarios,
these vineyard instances would typically be distributed across multiple machines within
the cluster.

.. code:: console

    $ python3 -m vineyard --socket /var/run/vineyard.sock1
    $ python3 -m vineyard --socket /var/run/vineyard.sock2

With this setup, we can create a distributed pair of arrays in vineyard, where the first
array is stored in the first vineyard instance (listening to ipc_socket at `/var/run/vineyard.sock1`),
and the second array is stored in the second instance (listening to ipc_socket at
`/var/run/vineyard.sock2`).

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

In this example, we can access the metadata of the pair object from `client2` even
though it was created by `client1`. However, we cannot retrieve the payload of the
first element of the pair from `client2`, as it is stored locally within the first
vineyard instance.

Creating Builders and Resolvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As demonstrated in :ref:`builder-resolver`, vineyard enables users to register
builders and resolvers for constructing and resolving vineyard objects from/to
client-side data types based on specific computational requirements.

For instance, if we use ``pyarrow`` types in our context, we can define the builder and
resolver for the conversion between ``vineyard::NumericArray`` and ``pyarrow.NumericArray``
as follows:

.. code:: python

    >>> def numeric_array_builder(client, array, builder):
    >>>     meta = ObjectMeta()
    >>>     meta['typename'] = 'vineyard::NumericArray<%s>' % array.type
    >>>     meta['length_'] = len(array)
    >>>     meta['null_count_'] = array.null_count
    >>>     meta['offset_'] = array.offset
    >>>
    >>>     null_bitmap = buffer_builder(client, array.buffers()[0], builder)
    >>>     buffer = buffer_builder(client, array.buffers()[1], builder)
    >>>
    >>>     meta.add_member('buffer_', buffer)
    >>>     meta.add_member('null_bitmap_', null_bitmap)
    >>>     meta['nbytes'] = array.nbytes
    >>>     return client.create_metadata(meta)

    >>> def numeric_array_resolver(obj):
    >>>     meta = obj.meta
    >>>     typename = obj.typename
    >>>     value_type = normalize_dtype(re.match(r'vineyard::NumericArray<([^>]+)>', typename).groups()[0])
    >>>     dtype = pa.from_numpy_dtype(value_type)
    >>>     buffer = as_arrow_buffer(obj.member('buffer_'))
    >>>     null_bitmap = as_arrow_buffer(obj.member('null_bitmap_'))
    >>>     length = int(meta['length_'])
    >>>     null_count = int(meta['null_count_'])
    >>>     offset = int(meta['offset_'])
    >>>     return pa.lib.Array.from_buffers(dtype, length, [null_bitmap, buffer], null_count, offset)

Finally, we register the builder and resolver for automatic building and resolving:
.. code:: python

    >>> builder_ctx.register(pa.NumericArray, numeric_array_builder)
    >>> resolver_ctx.register('vineyard::NumericArray', numeric_array_resolver)

In some cases, we may have multiple resolvers or builders for a specific type.
For instance, the `vineyard::Tensor` object can be resolved as either `numpy.ndarray` or
`xgboost::DMatrix`. To accommodate this, we could have:

.. code:: python

    >>> resolver_ctx.register('vineyard::Tensor', numpy_resolver)
    >>> resolver_ctx.register('vineyard::Tensor', xgboost_resolver)

This flexibility enables seamless integration with various libraries and frameworks by
effectively handling different data types and their corresponding resolvers or builders.

.. code:: python

    def xgboost_resolver(obj):
        ...

    default_resolver_context.register('vineyard::Tensor', xgboost_resolver)

at the same time. The stackable :code:`resolver_context` could help there,

.. code:: python

    with resolver_context({'vineyard::Tensor', xgboost_resolver}):
        ...

Assuming the default context resolves `vineyard::Tensor` to `numpy.ndarray`, the
`with resolver_context` allows for temporary resolution of `vineyard::Tensor` to
`xgboost::DMatrix`. Upon exiting the context, the global environment reverts to
its default state.

The `with resolver_context` can be nested for additional flexibility.
