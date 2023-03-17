:code:`multiprocessing.shared_memory` in Python
===============================================

Vineyard supports shared memory interface of :class:`SharedMemory` and
:class:`ShareableList` and the API is compatible with `multiprocessing.shared_memory`_.

The shared memory interface can be used in the following way:

.. code:: python

     >>> from vineyard import shared_memory
     >>> value = shared_memory.ShareableList(client, [b"a", "bb", 1234, 56.78, True])
     >>> value
     ShareableList([b'a', 'bb', 1234, 56.78, True], name='o8000000119aa10c0')
     >>> value[4] = False
     >>> value
     ShareableList([b'a', 'bb', 1234, 56.78, False], name='o8000000119aa10c0')

.. caution::

   Note that the semantic of the vineyard's :code:`shared_memory` is slightly different
   with the :code:`shared_memory` in python's multiprocessing module. Shared memory in
   vineyard cannot be mutable after been visible to other clients.

We have added a :code:`freeze` method to make such transformation happen:

.. code:: python

     >>> value.freeze()

After being freezed, the shared memory (aka. the :code:`ShareableList` in this case)
is available for other clients:

.. code:: python

     >>> value1 = shared_memory.ShareableList(client, name=value.shm.name)
     >>> value1
     ShareableList([b'a', 'bb', 1234, 56.78, False], name='o8000000119aa10c0')

For more details, see :ref:`shared-memory`.

.. _multiprocessing.shared_memory: https://docs.python.org/3/library/multiprocessing.shared_memory.html
