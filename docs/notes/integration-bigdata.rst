Big-data on Vineyard
====================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   integration/dask.rst
   integration/ml.rst
   integration/ray.rst

Vineyard is designed for serving as the immediate data sharing engine and has
been integrated with various big-data computing engines. Namely the machine
learning frameworks as well as the distributed data processing engine Dask.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: integration/ml
      :type: ref
      :text: Machine Learning
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Executing machine learning workflows on top of vineyard.

   ---

   .. link-button:: integration/dask
      :type: ref
      :text: Dask
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Using vineyard as the data source / sink of dask computations.

   ---

   .. link-button:: integration/dask
      :type: ref
      :text: Dask
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Using vineyard as the data source / sink of Ray computations.
