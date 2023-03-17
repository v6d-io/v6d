Getting Started
===============

.. _getting-started:

Installing vineyard
-------------------

Vineyard is distributed as a `Python package`_ and can be easily installed with :code:`pip`:

.. code:: console

   $ pip3 install vineyard

Launching vineyard server
-------------------------

.. code:: console

   $ python3 -m vineyard

A vineyard daemon server will be launched with default settings. By default, :code:`/var/run/vineyard.sock`
will be used by vineyardd to listen for incoming IPC connections.

To stop running the vineyardd instance, you can press :code:`Ctrl-C` in the terminal.

.. tip::

   If you encounter errors like ``cannot launch vineyardd on '/var/run/vineyard.sock':
   Permission denied,``, that means **you don't have the permission** to create a UNIX-domain
   socket at :code:`/var/run/vineyard.sock`, you could either

   - run vineyard as root, using ``sudo``:

     .. code:: console

        $ sudo -E python3 -m vineyard

   - or, change the socket path to a writable location with the ``--socket`` command
     line option:

     .. code:: console

        $ python3 -m vineyard --socket /tmp/vineyard.sock

Connecting to vineyard
----------------------

Once launched, you could call :code:`vineyard.connect` with the socket name to start a vineyard client
from Python:

.. code:: python

   >>> import vineyard
   >>> client = vineyard.connect('/var/run/vineyard.sock')

Putting and getting Python objects
----------------------------------

Vineyard is designed as an in-memory object store and provides two high-level APIs :code:`put` and
:code:`get` for creating and accessing shared objects to seamlessly interoperate with the Python
ecosystem. The former returns a :code:`vineyard.ObjectID` when succeed and it can be further used
to retrieve shared objects from vineyard by the latter.

In following example, We first use :code:`client.put()` to build the vineyard object from the numpy
ndarray ``arr``, which returns the ``object_id`` that is the unique id in vineyard to represent
the object. Given the ``object_id``, we can obtain a shared-memory object from vineyard with method
:code:`client.get()`.

.. code:: python

   >>> import numpy as np
   >>>
   >>> object_id = client.put(np.random.rand(2, 4))
   >>> object_id
   o0015c78883eddf1c
   >>>
   >>> shared_array = client.get(object_id)
   >>> shared_array
   ndarray([[0.39736989, 0.38047846, 0.01948815, 0.38332264],
            [0.61671189, 0.48903213, 0.03875045, 0.5873005 ]])

.. note::

   :code:`shared_array` doesn't allocate extra memory in the Python process; instead, it shares memory
   with the vineyard server via `mmap`_ and the process is zero-copy.

The sharable objects can be complex and nested. Like numpy ndarray, the pandas dataframe ``df`` can
be seamlessly put into vineyard and get back with the ``.put()`` and ``.get()`` method as follows,

.. code:: python

   >>> import pandas as pd
   >>>
   >>> df = pd.DataFrame({'u': [0, 0, 1, 2, 2, 3],
   >>>                    'v': [1, 2, 3, 3, 4, 4],
   >>>                    'weight': [1.5, 3.2, 4.7, 0.3, 0.8, 2.5]})
   >>> object_id = client.put(df)
   >>>
   >>> shared_dataframe = client.get(object_id)
   >>> shared_dataframe
      u  v  weight
   0  0  1     1.5
   1  0  2     3.2
   2  1  3     4.7
   3  2  3     0.3
   4  2  4     0.8
   5  3  4     2.5

Under the hood, vineyard implements a builder/resolver mechanism to represent arbitrary
data structure as *vineyard objects* and resolve back to native values in the corresponding
programming languages and computing systems, see also :ref:`divein-driver-label`.

Sharing objects between tasks
-----------------------------

Vineyard is designed for sharing intermediate data between tasks. The following example
demonstrates how dataframe can be passed between two **processes** using vineyard, namely
producer and consumer in the following example:

.. code:: python

   import multiprocessing as mp
   import vineyard

   import numpy as np
   import pandas as pd

   socket = '/var/run/vineyard.sock'

   def produce(name):
      client = vineyard.connect(socket)
      client.put(pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD')),
                 persist=True, name=name)

   def consume(name):
      client = vineyard.connect(socket)
      print(client.get(name=name).sum())

   if __name__ == '__main__':
      name = 'dataset'

      producer = mp.Process(target=produce, args=(name,))
      producer.start()
      consumer = mp.Process(target=consume, args=(name,))
      consumer.start()

      producer.join()
      consumer.join()

Running the code above, you should see the following output:

.. code:: python

   A   -4.529080
   B   -2.969152
   C   -7.067356
   D    4.003676
   dtype: float64

Next steps
----------

Beyond the core functionality of sharing objects between tasks, vineyard also provides

- Distributed objects and stream abstraction over immutable chunks;
- An IDL (:ref:`vcdl`) that helps integrate vineyard with other systems at the minimalist cost;
- A mechanism of pluggable drivers for miscellaneous tasks that serve as the glue
  between the core compute engine and the external world, e.g., data sources, data
  sinks;
- Integration with Kubernetes for sharing between tasks in workflows that deployed
  on cloud-native infrastructures.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: architecture
      :type: ref
      :text: Architecture
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Overview of vineyard.

Learn more about vineyard's key concepts from the following user guides:

.. panels::
   :header: text-center
   :container: container-lg pb-4
   :column: col-lg-4 col-md-4 col-sm-4 col-xs-12 p-2
   :body: text-center

   .. link-button:: key-concepts/objects
      :type: ref
      :text: Vineyard Objects
      :classes: btn-block stretched-link

   Illustrate the design of object model in vineyard.

   ---

   .. link-button:: key-concepts/vcdl
      :type: ref
      :text: VCDL
      :classes: btn-block stretched-link

   How vineyard been integrated with other computing systems?

   ---

   .. link-button:: key-concepts/io-drivers
      :type: ref
      :text: I/O Drivers
      :classes: btn-block stretched-link

   Design and implementation of the pluggable routines for I/O, repartition, migration, etc.

Vineyard is natural fit to cloud-native computing, where vineyard can be deployed and
managed the *vineyard operator*, and provides data-aware scheduling for data analytical
workflows to archive efficient data sharing on Kubernetes. More details about vineyard
on Kubernetes can be found from

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: cloud-native/deploy-kubernetes
      :type: ref
      :text: Kubernetes
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Deploy vineyard on Kubernetes and accelerating your big-data workflows.

.. _Python package: https://pypi.org/project/vineyard
.. _mmap: https://man7.org/linux/man-pages/man2/mmap.2.html
