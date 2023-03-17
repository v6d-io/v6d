Workflow orchestration
======================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   integration/airflow.rst

Vineyard seamlessly integrates with the workflow orchestration engine, Apache Airflow,
enabling users to effortlessly incorporate Vineyard into their workflows for enhanced performance.
Moreover, the Airflow integration empowers users to work with large Python objects
featuring complex data types (e.g., :code:`pandas.DataFrame`) at minimal cost, while
eliminating the need for cumbersome :code:`pickle.dump/loads` operations.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: integration/airflow
      :type: ref
      :text: Airflow
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Airflow uses vineyard as the XCom backend to efficiently handle complex data in Python.
