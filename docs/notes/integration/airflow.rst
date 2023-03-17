Airflow on Vineyard
===================

Big data analytical pipelines often involve various types of workloads, each
requiring a dedicated computing system to complete the task. Intermediate
data flows between tasks in the pipeline, and the additional cost of transferring data
accounts for a significant portion of the end-to-end performance in real-world deployments,
making optimization a challenging task.

Integrating Vineyard with Airflow presents opportunities to alleviate this problem.

Introducing Airflow
-------------------

Airflow is a platform that enables users to programmatically author, schedule, and
monitor workflows. Users organize tasks in a Directed Acyclic Graph (DAG), and the
Airflow scheduler executes the tasks on workflows while adhering to the specified
dependencies.

Consider the following ETL workflow as an example [1]_,

.. code:: python

    @dag(schedule_interval=None, start_date=days_ago(2), tags=['example'])
    def tutorial_taskflow_api_etl():
        @task()
        def extract():
            data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'

            order_data_dict = json.loads(data_string)
            return order_data_dict

        @task(multiple_outputs=True)
        def transform(order_data_dict: dict):
            return {"total_order_value": total_order_value}

        @task()
        def load(total_order_value: float):
            print(f"Total order value is: {total_order_value:.2f}")

        order_data = extract()
        order_summary = transform(order_data)


    tutorial_etl_dag = tutorial_taskflow_api_etl()

It forms the following DAG, including three individual tasks as the nodes, and
runs the tasks sequentially based on their data This forms a DAG, including
three individual tasks as nodes, and edges between nodes that describe the
data dependency relations. The Airflow scheduler runs the tasks sequentially
based on their data dependencies.dependencies. Airflow ETL Workflow

Airflow on Vineyard
-------------------

The Rationale for Airflow on Vineyard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Airflow excels at defining and orchestrating complex workflows. However, managing
data flow within the pipeline remains a challenge. Airflow relies on database
backends such as SQLite, MySQL, and PostgreSQL to store intermediate data between
tasks. In real-world scenarios, large-scale data, such as large tensors, dataframes,
and distributed graphs, cannot fit into these databases. As a result, external
storage systems like HDFS and S3 are used to store intermediate data, with only
an identifier stored in the database.

Utilizing external storage systems to share intermediate data among tasks in big
data analytical pipelines incurs performance costs due to data copying,
serialization/deserialization, and network data transfer.

Vineyard is designed to efficiently share intermediate in-memory data for big data
analytical pipelines, making it a natural fit for workloads on Airflow.

How Vineyard Enhances Airflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Airflow allows users to register an external **XCom** backend, which is precisely
what Vineyard is designed for.

Vineyard serves as an *XCom* backend for Airflow workers, enabling the transfer of
large-scale data objects between tasks without relying on Airflow's database backend
or external storage systems like HDFS. The Vineyard XCom backend also handles object
migration when the required inputs are not located where the task is scheduled to
execute.

Vineyard's XCom backend achieves its functionality by injecting hooks into the
processes of saving values to the backend and fetching values from the backend,
as described below:

.. code:: python

    class VineyardXCom(BaseXCom):

        @staticmethod
        def serialize_value(value: Any):
            """ Store the value to vineyard server, and serialized the result
                Object ID to save it into the backend database later.
            """

        @staticmethod
        def deserialize_value(result: "XCom") -> Any:
            """ Obtain the Object ID after deserialization, and fetching the
                underlying value from vineyard.

                This value is resolved from vineyard objects in a zero-copy
                fashion.
            """


Addressing Distributed Deployment Challenges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Airflow supports parallel task execution across multiple workers to efficiently
process complex workflows. In a distributed deployment (using the `CeleryExecutor`),
tasks sharing intermediate data might be scheduled on different workers, necessitating
remote data access.

Vineyard seamlessly handles object migration for various data types. In the XCom backend,
when the IPC client encounters remote objects, it triggers a migration action to move
the objects to the local worker, ensuring input data is readily available before task
execution.

This transparent object migration simplifies complex data operations and movement,
allowing data scientists to focus on computational logic when developing big data
applications on Airflow.

Running Vineyard + Airflow
--------------------------

Users can try Airflow provider for Vineyard by the following steps:

1. Install required packages:

   .. code:: bash

       pip3 install airflow-provider-vineyard

2. Configure Vineyard locally

   The vineyard server can be easier launched locally with the following command:

   .. code:: bash

       python -m vineyard --socket=/tmp/vineyard.sock

   See also our documentation about `launching vineyard`_.

3. Configure Airflow to use the vineyard XCom backend by specifying the environment
   variable

   .. code:: bash

       export AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom

   and configure the location of UNIX-domain IPC socket for vineyard client by

   .. code:: bash

       export AIRFLOW__VINEYARD__IPC_SOCKET=/tmp/vineyard.sock

   or

   .. code:: bash

       export VINEYARD_IPC_SOCKET=/tmp/vineyard.sock

4. Launching your airflow scheduler and workers, and run the following DAG as example,

   .. code:: python

       import numpy as np
       import pandas as pd

       from airflow.decorators import dag, task
       from airflow.utils.dates import days_ago

       default_args = {
           'owner': 'airflow',
       }

       @dag(default_args=default_args, schedule_interval=None, start_date=days_ago(2), tags=['example'])
       def taskflow_etl_pandas():
           @task()
           def extract():
               order_data_dict = pd.DataFrame({
                   'a': np.random.rand(100000),
                   'b': np.random.rand(100000),
               })
               return order_data_dict

           @task(multiple_outputs=True)
           def transform(order_data_dict: dict):
               return {"total_order_value": order_data_dict["a"].sum()}

           @task()
           def load(total_order_value: float):
               print(f"Total order value is: {total_order_value:.2f}")

           order_data = extract()
           order_summary = transform(order_data)
           load(order_summary["total_order_value"])

       taskflow_etl_pandas_dag = taskflow_etl_pandas()

In the example above, the `extract` and `transform` tasks share a `pandas.DataFrame` as
intermediate data. This presents a challenge, as the DataFrame cannot be pickled, and when
dealing with large data, it cannot fit into the backend databases of Airflow.

This example is adapted from the Airflow documentation. For more information, refer to the
`Tutorial on the Taskflow API`_.

Further Ahead
-------------

The Airflow provider for Vineyard, currently in its experimental stage, demonstrates
significant potential for efficiently and flexibly sharing large-scale intermediate data
in big data analytical workflows within Airflow.

The Airflow community is actively working to enhance support for modern big data and AI
applications. We believe that the integration of Vineyard, Airflow, and other cloud-native
infrastructures can provide a more effective and efficient solution for data scientists.


.. [1] See: https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html

.. _launching vineyard: https://v6d.io/notes/getting-started.html#starting-vineyard-server
.. _Tutorial on the Taskflow API: https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
