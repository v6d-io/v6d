Workflow orchestration
======================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   integration/airflow.rst
   integration/kedro.md

Vineyard seamlessly integrates with the workflow orchestration engines, e.g.,
Apache Airflow and Kedro, enabling users to effortlessly incorporate Vineyard
into their workflows for enhanced performance.

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

The Kedro integration enables users to easily share large data objects across
nodes in a pipeline and eliminates the high cost of (de)serialization and I/O
compared with alternatives like AWS S3 or Minio, without the need to modify
the pipeline code intrusively, and provides seamless user experience when scaling
pipelines to Kubernetes.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: integration/kedro
      :type: ref
      :text: Kedro
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Kedro uses vineyard as a `DataSet` implementation for efficient intermediate data sharing.
