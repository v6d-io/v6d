
Data processing
===============

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   ./data-processing/using-objects-python.rst
   ./data-processing/python-sharedmemory.rst
   ./data-processing/distributed-learning.rst
   ./data-processing/accelerate-data-sharing-in-kedro.rst
   ./data-processing/gpu-memory-sharing.rst

In these comprehensive case studies, we demonstrate how to seamlessly integrate vineyard's
capabilities with existing data-intensive tasks. By incorporating vineyard into complex
workflows involving multiple computing engines, users can experience significant
improvements in both performance and ease of use.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./data-processing/using-objects-python
      :type: ref
      :text: Python Objects
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Effortlessly share Python objects between processes using vineyard's intuitive and efficient approach.

   ---

   .. link-button:: ./data-processing/python-sharedmemory
      :type: ref
      :text: SharedMemory in Python
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Utilize vineyard as an elegant alternative to :code:`multiprocessing.shared_memory` in Python.

   ---

   .. link-button:: ./data-processing/distributed-learning
      :type: ref
      :text: Distributed Learning
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Discover how vineyard enhances distributed machine learning training workflows by
   seamlessly integrating with various computing engines for improved efficiency and elegance.

   ---

   .. link-button:: ./data-processing/accelerate-data-sharing-in-kedro
      :type: ref
      :text: Accelerate Data Sharing in Kedro
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Vineyard serves as the :code:`DataSet` backend for Kedro pipelines, enabling
   efficient data sharing between tasks without intrusive code modification, even
   when the pipeline is deployed to Kubernetes.

   ---

   .. link-button:: ./data-processing/gpu-memory-sharing
      :type: ref
      :text: GPU Memory Sharing
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Vineyard supports sharing GPU memory in zero-copy manner, enabling efficient data sharing
   between GPU-accelerated tasks.
