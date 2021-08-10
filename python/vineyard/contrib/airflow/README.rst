Apache Airflow Provider for Vineyard
====================================

The apache airflow provider for vineyard contains components to sharing intermediate
data among tasks in Airflow workflows using Vineyard.

Vineyard works as a *XCom* backend for airflow workers to allow transferring
large-scale data objects between tasks that cannot be fit into the Airflow's
database backend without involving external storage systems like HDFS. The
Vineyard XCom backend handles object migration as well when the required inputs
is not located on where the task is scheduled to execute.

Requirements
------------

The following packages are needed to run Airflow on Vineyard,

- airflow >= 2.1.0
- vineyard >= 0.2.7

Configuration and Usage
-----------------------

1. Install required packages:

    .. code:: bash

        pip3 install airflow-provider-vineyard

2. Configure Vineyard locally

    The vineyard server can be easier launched locally with the following command:

    .. code:: bash

        vineyardd --socket=/tmp/vineyard.sock

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
                    'b': np.random.rand(100000)
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

In above example, task :code:`extract` and task :code:`transform` shares a
:code:`pandas.DataFrame` as the intermediate data, which is impossible as
it cannot be pickled and when the data is large, it cannot be fit into the
table in backend databases of Airflow.

The example is adapted from the documentation of Airflow, see also
`Tutorial on the Taskflow API`_.


.. _launching vineyard: https://v6d.io/notes/getting-started.html#starting-vineyard-server
.. _Tutorial on the Taskflow API: https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
