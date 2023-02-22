Apache Airflow Provider for Vineyard
====================================

The apache airflow provider for vineyard contains components to sharing intermediate
data among tasks in Airflow workflows using Vineyard.

Vineyard works as a *XCom* backend for airflow workers to allow transferring
large-scale data objects between tasks that cannot be fit into the Airflow's
database backend without involving external storage systems like HDFS. The
Vineyard XCom backend handles object migration as well when the required inputs
is not located on where the task is scheduled to execute.

Table of Contents
-----------------

- [Requirements](#requirements)
- [Configuration and Usage](#configuration-and-usage)
- [Run the tests](#run-tests)
- [Deploy using Docker Compose](#deploy-using-docker-compose)
- [Deploy on Kubernetes](#deploy-on-kubernetes)

Requirements <a name="requirements"/>
------------

The following packages are needed to run Airflow on Vineyard,

- airflow >= 2.1.0
- vineyard >= 0.2.12

Configuration and Usage <a name="configuration-and-usage"/>
-----------------------

1. Install required packages:

        pip3 install airflow-provider-vineyard

2. Configure Vineyard locally

    The vineyard server can be easier launched locally with the following command:

        python3 -m vineyard --socket=/tmp/vineyard.sock

    See also our documentation about [launching vineyard][1].

3. Configure Airflow to use the vineyard XCom backend by specifying the environment
    variable

        export AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom

    and configure the location of UNIX-domain IPC socket for vineyard client by

        export AIRFLOW__VINEYARD__IPC_SOCKET=/tmp/vineyard.sock

    or

        export VINEYARD_IPC_SOCKET=/tmp/vineyard.sock

4. Launching your airflow scheduler and workers, and run the following DAG as example,

        ```python
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
        ```

In above example, task :code:`extract` and task :code:`transform` shares a
:code:`pandas.DataFrame` as the intermediate data, which is impossible as
it cannot be pickled and when the data is large, it cannot be fit into the
table in backend databases of Airflow.

The example is adapted from the documentation of Airflow, see also
[Tutorial on the Taskflow API][2].

Run the tests <a name="run-tests"/>
-------------

1. Start your vineyardd with the following command,

        python3 -m vineyard

2. Set airflow to use the vineyard XCom backend, and run tests with pytest,

        export AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom

        pytest -s -vvv python/vineyard/contrib/airflow/tests/test_python_dag.py
        pytest -s -vvv python/vineyard/contrib/airflow/tests/test_pandas_dag.py


The pandas test suite is not possible to run with the default XCom backend, vineyard
enables airflow to exchange **complex** and **big** data without modify the DAG and tasks!

Deploy using Docker Compose <a name="deploy-using-docker-compose"/>
---------------------------

We provide a reference docker compose settings (see [docker-compose.yaml](./docker/docker-compose.yaml))
for deploying airflow with vineyard as the XCom backend on Docker Compose.

The docker compose containers cloud be deployed as

```bash
$ cd docker/
$ docker compose up
```

We have also included a diff file [docker-compose.yaml.diff](./docker/docker-compose.yaml.diff) that shows
the changed pieces that can be introduced into your own docker compose deployment.

Deploy on Kubernetes <a name="deploy-on-kubernetes"/>
--------------------

We provide a reference settings (see [values.yaml](./values.yaml)) for deploying
airflow with vineyard as the XCom backend on Kubernetes, based on [the official helm charts][3].

Deploying vineyard requires etcd, to ease to deploy process, you first need to
setup a standalone etcd cluster. A _test_ etcd cluster with only one instance can
be deployed by

```bash
$ kubectl create -f etcd.yaml
```

The [values.yaml](./values.yaml) mainly tweaks the following settings:

- Installing vineyard dependency to the containers using pip before start workers
- Adding a vineyardd container to the airflow pods
- Mounting the vineyardd's UNIX-domain socket and shared memory to the airflow worker pods

Note that **the `values.yaml` may doesn't work in your environment**, as airflow requires
other settings like postgresql database, persistence volumes, etc. You can combine
the reference `values.yaml` with your own specific Airflow settings.

The [values.yaml](./values.yaml) for Airflow's helm chart can be used as

```bash
# add airflow helm stable repo
$ helm repo add apache-airflow https://airflow.apache.org
$ helm repo update

# deploy airflow
$ helm install -f values.yaml $RELEASE_NAME apache-airflow/airflow --namespace $NAMESPACE
```

[1]: https://v6d.io/notes/getting-started.html#starting-vineyard-server
[2]: https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
[3]: https://github.com/apache/airflow/tree/main/chart
