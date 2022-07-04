Integrations
============

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   notes/dask.rst
   notes/ml.rst
   notes/airflow.rst

Vineyard is designed for serving as the immediate data sharing engine and has
been integrated with various big-data computing engines. Namely the machine
learning frameworks as well as the distributed data processing engine Dask.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: notes/ml
      :type: ref
      :text: Machine Learning
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Executing machine learning workflows on top of vineyard.

   ---

   .. link-button:: notes/dask
      :type: ref
      :text: Dask
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Using vineyard as the data source / sink of dask computations.

Vineyard has integrated to the workflow orchestrating engines (apache airflow)
to helps use adopt vineyard into their own workflows for the performance gains.
Moreover, the airflow integration allows user operating on large Python objects
of complex data types (e.g., :code:`pandas.DataFrame`) at low cost and avoid
the burden of :code:`pickle.dump/loads`.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: notes/airflow
      :type: ref
      :text: Airflow
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Airflow uses vineyard as the XCom backend to efficiently handle complex data in Python.
