.. _define-python-types:

Define Data Types in Python
---------------------------

Objects
^^^^^^^

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

Builders and resolvers
^^^^^^^^^^^^^^^^^^^^^^

As we shown in :ref:`builder-resolver`, vineyard allows users to register
builders/resolvers to build/resolve vineyard objects from/to the data types
in the client side based on the computation requirements.

Suppose ``pyarrow`` types are employed in the context, then we can define the builder and
resolver between ``vineyard::NumericArray`` and ``pyarrow.NumericArray`` as follows:

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

There are cases where we have more than one resolvers or builders for a certain type,
e.g., the :code:`vineyard::Tensor` object can be resolved as :code:`numpy.ndarray` or
:code:`xgboost::DMatrix`. We could have

.. code:: python

    def numpy_resolver(obj):
        ...

    default_resolver_context.register('vineyard::Tensor', numpy_resolver)

and

.. code:: python

    def xgboost_resolver(obj):
        ...

    default_resolver_context.register('vineyard::Tensor', xgboost_resolver)

at the same time. The stackable :code:`resolver_context` could help there,

.. code:: python

    with resolver_context({'vineyard::Tensor', xgboost_resolver}):
        ...

Assuming the default context resolves :code:`vineyard::Tensor` to :code:`numpy.ndarray`,
inside the :code:`with resolver_context` the :code:`vineyard::Tensor` will be resolved
to :code:`xgboost::DMatrix`, and after exiting the context the global environment
will be restored back as default.

The :code:`with resolver_context` is nestable as well.
