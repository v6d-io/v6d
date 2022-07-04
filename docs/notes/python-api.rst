Python API Reference
====================

.. _python-api:

.. default-domain:: py

.. currentmodule:: vineyard

Objects
-------

.. autoclass:: ObjectID
    :special-members:
    :members:

.. autoclass:: Object
    :members:

.. autoclass:: ObjectBuilder
    :members:

Metadata
--------

.. autoclass:: ObjectMeta
    :special-members:
    :members:

Vineyard Client
---------------

.. autofunction:: connect

.. autoclass:: IPCClient
    :inherited-members:
    :members:

.. autoclass:: RPCClient
    :inherited-members:
    :members:

Vineyard Cluster
----------------

.. autoclass:: InstanceStatus
    :special-members:
    :members:

Blob
----

.. autoclass:: Blob
    :members:

.. autoclass:: BlobBuilder
    :members:

Resolvers and Builders
----------------------

.. autoclass:: vineyard.core.resolver.ResolverContext
    :members:

.. autofunction:: vineyard.core.resolver.get_current_resolvers
.. autofunction:: vineyard.core.resolver.resolver_context

.. autoclass:: vineyard.core.builder.BuilderContext
    :members:

.. autofunction:: vineyard.core.builder.get_current_builders
.. autofunction:: vineyard.core.builder.builder_context

.. autoclass:: vineyard.core.driver.DriverContext
    :members:

.. autofunction:: vineyard.core.driver.get_current_drivers
.. autofunction:: vineyard.core.driver.driver_context

.. _shared-memory:

Shared Memory
-------------

.. autoclass:: vineyard.shared_memory.SharedMemory
    :members:

.. autoclass:: vineyard.shared_memory.ShareableList
    :members:

Deployment
----------

.. autofunction:: vineyard.init
.. autofunction:: vineyard.shutdown
.. autofunction:: vineyard.get_current_client
.. autofunction:: vineyard.get_current_socket
.. autofunction:: vineyard.deploy.local.start_vineyardd
.. autofunction:: vineyard.deploy.distributed.start_vineyardd

I/O Drivers
-----------

.. autofunction:: vineyard.io.open
.. autofunction:: vineyard.io.read
.. autofunction:: vineyard.io.write
