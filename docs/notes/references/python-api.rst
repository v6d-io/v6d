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

Vineyard client
---------------

.. autofunction:: connect

.. autoclass:: IPCClient
    :inherited-members:
    :members:

.. autoclass:: RPCClient
    :inherited-members:
    :members:

Vineyard cluster
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

.. autoclass:: RemoteBlob
    :members:

.. autoclass:: RemoteBlobBuilder
    :members:

Resolvers and builders
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

Shared memory
-------------

.. autoclass:: vineyard.shared_memory.SharedMemory
    :members:

.. autoclass:: vineyard.shared_memory.ShareableList
    :members:

.. _vineyard-python-deployment-api:

Deployment
----------

.. autofunction:: vineyard.connect
.. autofunction:: vineyard.get_current_socket
.. autofunction:: vineyard.deploy.local.start_vineyardd
.. autofunction:: vineyard.deploy.distributed.start_vineyardd
.. autofunction:: vineyard.deploy.kubernetes.start_vineyardd
.. autofunction:: vineyard.deploy.kubernetes.delete_kubernetes_objects

I/O Drivers
-----------

.. autofunction:: vineyard.io.open
.. autofunction:: vineyard.io.read
.. autofunction:: vineyard.io.write

.. _python-api-streams:

Streams
-------

.. autoclass:: vineyard.io.byte.ByteStream
    :members:

.. autoclass:: vineyard.io.dataframe.DataframeStream
    :members:

.. autoclass:: vineyard.io.recordbatch.RecordBatchStream
    :members:

Interacting with the CSI Driver
-------------------------------

.. autofunction:: vineyard.csi.read
.. autofunction:: vineyard.csi.write

Vineyard Cli Tool
-----------------

You can also use the Python API to interact with internal
`Vineyard Cli Tool`_.

.. code-block:: bash

    $ python -m vineyard.cli <command> [options]

Vineyard LLM KV Cache
---------------------

Before using the KV Cache, you need to install the ``vineyard-llm`` package.

.. autoclass:: vineyard.llm.KV_Cache
    :special-members:
    :members:

.. _Vineyard Cli Tool: https://v6d.io/notes/cloud-native/vineyardctl.html
