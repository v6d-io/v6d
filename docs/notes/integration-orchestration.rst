Workflow orchestration
======================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   integration/airflow.rst

Vineyard has integrated to the workflow orchestrating engines (apache airflow)
to helps use adopt vineyard into their own workflows for the performance gains.
Moreover, the airflow integration allows user operating on large Python objects
of complex data types (e.g., :code:`pandas.DataFrame`) at low cost and avoid
the burden of :code:`pickle.dump/loads`.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: integration/airflow
      :type: ref
      :text: Airflow
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Airflow uses vineyard as the XCom backend to efficiently handle complex data in Python.
