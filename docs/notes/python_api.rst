Python API Reference
====================

.. default-domain:: py

.. _python_api:

.. contents::
    :local:

.. currentmodule:: vineyard

ObjectID and Object
-------------------

.. autoclass:: ObjectID
    :special-members:
    :members:

.. autoclass:: Object
    :members:

.. autoclass:: ObjectBuilder
    :members:

Metadata of objects
-------------------

.. autoclass:: ObjectMeta
    :special-members:
    :members:

Connect to vineyard
-------------------

.. autofunction:: connect

.. autoclass:: IPCClient
    :inherited-members:
    :members:

.. autoclass:: RPCClient
    :inherited-members:
    :members:

State of server
---------------

.. autoclass:: InstanceStatus
    :special-members:
    :members:

Primitives
----------

.. autoclass:: Blob
    :members:

.. autoclass:: BlobBuilder
    :members:

.. _shared-memory:

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

Shared Memory
-------------

.. autoclass:: vineyard.shared_memory.SharedMemory
    :members:

.. autoclass:: vineyard.shared_memory.ShareableList
    :members:

Deployment
----------

.. autofunction:: vineyard.deploy.local.start_vineyardd
.. autofunction:: vineyard.deploy.distributed.start_vineyardd

IO Facilities
-------------

.. autofunction:: vineyard.io.open
.. autofunction:: vineyard.io.read
.. autofunction:: vineyard.io.write
