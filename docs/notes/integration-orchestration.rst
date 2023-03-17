Workflow orchestration
======================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   integration/airflow.rst

Vineyard has integrated with the workflow orchestration engine (Apache Airflow)
to help users adopt Vineyard into their own workflows for performance gains.
Furthermore, the Airflow integration allows users to operate on large Python objects
with complex data types (e.g., :code:`pandas.DataFrame`) at a low cost and avoid
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
