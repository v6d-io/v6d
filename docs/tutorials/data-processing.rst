
Data processing
===============

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   ./data-processing/using-objects-python.rst
   ./data-processing/python-sharedmemory.rst
   ./data-processing/distributed-learning.rst

We showcase step-by-step case studies of how to combine the functionalities of vineyard
with existing data-intensive jobs. We show that vineyard can bring huge gains in both
performance and conveniences when users have a complex workflow that involves multiple
computing engines.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./data-processing/using-objects-python
      :type: ref
      :text: Python Objects
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Sharing Python objects between processes using vineyard is as simple as 123.

   ---

   .. link-button:: ./data-processing/python-sharedmemory
      :type: ref
      :text: SharedMemory in Python
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Using vineyard as using :code:`multiprocessing.shared_memory` in Python.

   ---

   .. link-button:: ./data-processing/distributed-learning
      :type: ref
      :text: Distributed Learning
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   How vineyard can help in a distributed machine learning training workflow where
   various computing engine are involved.
