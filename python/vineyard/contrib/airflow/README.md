Apache Airflow Provider for Vineyard
====================================

The apache airflow provider for vineyard contains components to share intermediate
data among tasks in Airflow workflows using vineyard.

Vineyard works as the *XCom* backend for airflow workers to allow transferring
large-scale data objects between tasks that cannot be fit into the Airflow's
database backend without involving external storage systems like HDFS. The
Vineyard XCom backend handles object migration as well when the required inputs
are not located where the task is scheduled to execute.

Table of Contents
-----------------

- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Run the tests](#run-tests)
- [Deploy using Docker Compose](#deploy-using-docker-compose)
- [Deploy on Kubernetes](#deploy-on-kubernetes)

Requirements <a name="requirements"/>
------------

The following packages are needed to run Airflow on Vineyard,

- airflow >= 2.1.0
- vineyard >= 0.2.12

Configuration <a name="configuration"/>
-------------

1. Install required packages:

        pip3 install airflow-provider-vineyard

2. Configure Vineyard locally

    The vineyard server can be easier launched locally with the following command:

        python3 -m vineyard --socket=/tmp/vineyard.sock

    See also our documentation about [launching vineyard][1].

3. Configure Airflow to use the vineyard XCom backend by specifying the environment
   variable

       export AIRFLOW__CORE__XCOM_BACKEND=vineyard.contrib.airflow.xcom.VineyardXCom

   and configure the location of the UNIX-domain IPC socket for vineyard client by

       export AIRFLOW__VINEYARD__IPC_SOCKET=/tmp/vineyard.sock

   or

       export VINEYARD_IPC_SOCKET=/tmp/vineyard.sock
    
   If you have deployed a distributed vineyard cluster, you can also specify the `persist` environment to enable the vineyard client to persist the data to the vineyard cluster.

       export AIRFLOW__VINEYARD__PERSIST=true

Usage <a name="usage"/>
-----

After installing the dependencies and preparing the vineyard server, you can
launch your airflow scheduler and workers, and run the following DAG as an example,

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

In the above example, task `extract` and task `transform` shares a
`pandas.DataFrame` as the intermediate data, which is impossible as
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

We provide a reference docker-compose settings (see [docker-compose.yaml](./docker/docker-compose.yaml))
for deploying airflow with vineyard as the XCom backend on Docker Compose. For more details, please refer to [the official documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html).

Before deploying the docker-compose, you need to initialize the environment as follows.

```bash
$ mkdir -p ./dags ./logs ./plugins ./config
$ echo -e "AIRFLOW_UID=$(id -u)" > .env
```

Then, copy the example dags to the `./dags` directory.

```bash
$ cp examples_dags/v6d*.py ./dags
```

After that, the docker-compose containers could be deployed as

```bash
$ cd docker/
$ docker build . -f Dockerfile -t vineyardcloudnative/vineyard-airflow:2.6.3
$ docker compose up
```

You can see the added DAGs and run them via [web ui](https://airflow.apache.org/docs/apache-airflow/stable/ui.html) or [cli](https://airflow.apache.org/docs/apache-airflow/stable/howto/usage-cli.html).

We have also included a diff file [docker-compose.yaml.diff](./docker/docker-compose.yaml.diff) that shows
the changed pieces that can be introduced into your own docker compose deployment.

Deploy on Kubernetes <a name="deploy-on-kubernetes"/>
--------------------

We provide a reference settings (see [values.yaml](./values.yaml)) for deploying
airflow with vineyard as the XCom backend on Kubernetes, based on [the official helm charts][3].

Next, we will show how to deploy airflow with vineyard on Kubernetes using the helm chart.

You are supposed to deploy the vineyard cluster on the
kubernetes cluster first. For example, you can deploy a vineyard cluster with the [vineyardctl command](https://github.com/v6d-io/v6d/tree/main/k8s/cmd#vineyardctl).

```bash
$ vineyardctl deploy vineyard-deployment --create-namespace
```

The [values.yaml](./values.yaml) mainly tweak the following settings:

- Installing vineyard dependency to the containers using pip before starting workers
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
$ helm install -f values.yaml airflow apache-airflow/airflow --namespace airflow --create-namespace
```

If you want to put the vineyard example DAGs into the airflow scheduler pod and the worker pod, you can use the following command.

```bash
$ kubectl cp ./example_dags/v6d_etl.py $(kubectl get pod -lcomponent=scheduler -n airflow -o jsonpath='{.items[0].metadata.name}'):/opt/airflow/dags -c scheduler -n airflow
$ kubectl cp ./example_dags/v6d_etl.py $(kubectl get pod -lcomponent=worker -n airflow -o jsonpath='{.items[0].metadata.name}'):/opt/airflow/dags -c worker -n airflow

```

[1]: https://v6d.io/notes/getting-started.html#starting-vineyard-server
[2]: https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html
[3]: https://github.com/apache/airflow/tree/main/chart
