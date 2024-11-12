:code:`multiprocessing.shared_memory` in Python
===============================================

Vineyard offers a shared memory interface through :class:`SharedMemory` and
:class:`ShareableList` classes, ensuring compatibility with Python's `multiprocessing.shared_memory`_.

Utilize the shared memory interface as demonstrated below:

.. code:: python

     >>> from vineyard import shared_memory
     >>> value = shared_memory.ShareableList(client, [b"a", "bb", 1234, 56.78, True])
     >>> value
     ShareableList([b'a', 'bb', 1234, 56.78, True], name='o8000000119aa10c0')
     >>> value[4] = False
     >>> value
     ShareableList([b'a', 'bb', 1234, 56.78, False], name='o8000000119aa10c0')

.. caution::

   Please be aware that the semantics of Vineyard's :code:`shared_memory` differ slightly
   from those of Python's multiprocessing module's :code:`shared_memory`. In Vineyard,
   shared memory cannot be modified once it becomes visible to other clients.

We have added a :code:`freeze` method to make such transformation happen:

.. code:: python

     >>> value.freeze()

After being frozen, the shared memory (aka. the :code:`ShareableList` in this case)
is available for other clients:

.. code:: python

     >>> value1 = shared_memory.ShareableList(client, name=value.shm.name)
     >>> value1
     ShareableList([b'a', 'bb', 1234, 56.78, False], name='o8000000119aa10c0')

For more details, see :ref:`shared-memory`.

.. _multiprocessing.shared_memory: https://docs.python.org/3/library/multiprocessing.shared_memory.html
