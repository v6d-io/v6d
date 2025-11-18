Kedro Vineyard Plugin
=====================

The Kedro vineyard plugin contains components (e.g., `DataSet` and `Runner`)
to share intermediate data among nodes in Kedro pipelines using vineyard.

Kedro on Vineyard
-----------------

Vineyard works as the *DataSet* provider for kedro workers to allow transferring
large-scale data objects between tasks that cannot be efficiently serialized and
is not suitable for `pickle`, without involving external storage systems like
AWS S3 (or Minio as an alternative). The Kedro vineyard plugin handles object migration
as well when the required inputs are not located where the task is scheduled to execute.

Requirements
------------

The following packages are needed to run Kedro on vineyard,

- kedro >= 0.18
- vineyard >= 0.14.5

Configuration
-------------

1. Install required packages:

       pip3 install vineyard-kedro

2. Configure Vineyard locally

    The vineyard server can be easier launched locally with the following command:

       python3 -m vineyard --socket=/tmp/vineyard.sock

    See also our documentation about [Launching Vineyard][1].

3. Configure the environment variable to tell Kedro vineyard plugin how to connect to the
   vineyardd server:

       export VINEYARD_IPC_SOCKET=/tmp/vineyard.sock

Usage
-----

After installing the dependencies and preparing the vineyard server, you can execute the
Kedro workflows as usual and benefits from vineyard for intermediate data sharing.

We take the [Iris example][2] as an example,

```bash
$ kedro new --starter=pandas-iris
```

The nodes in this pipeline look like

```python
def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_train = data.sample(
        frac=parameters["train_fraction"], random_state=parameters["random_state"]
    )
    data_test = data.drop(data_train.index)

    X_train = data_train.drop(columns=parameters["target_column"])
    X_test = data_test.drop(columns=parameters["target_column"])
    y_train = data_train[parameters["target_column"]]
    y_test = data_test[parameters["target_column"]]

    return X_train, X_test, y_train, y_test


def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> pd.Series:
    X_train_numpy = X_train.to_numpy()
    X_test_numpy = X_test.to_numpy()

    squared_distances = np.sum(
        (X_train_numpy[:, None, :] - X_test_numpy[None, :, :]) ** 2, axis=-1
    )
    nearest_neighbour = squared_distances.argmin(axis=0)
    y_pred = y_train.iloc[nearest_neighbour]
    y_pred.index = X_test.index

    return y_pred
```

You can see that the intermediate data between `split_data` and `make_predictions` is some pandas
dataframes and series.

Try running the pipeline without vineyard,

```bash
$ cd iris
$ kedro run
[05/25/23 11:38:56] INFO     Kedro project iris                                                                                       session.py:355
[05/25/23 11:38:57] INFO     Loading data from 'example_iris_data' (CSVDataSet)...                                               data_catalog.py:343
                    INFO     Loading data from 'parameters' (MemoryDataSet)...                                                   data_catalog.py:343
                    INFO     Running node: split: split_data([example_iris_data,parameters]) -> [X_train,X_test,y_train,y_test]          node.py:329
                    INFO     Saving data to 'X_train' (MemoryDataSet)...                                                         data_catalog.py:382
                    INFO     Saving data to 'X_test' (MemoryDataSet)...                                                          data_catalog.py:382
                    INFO     Saving data to 'y_train' (MemoryDataSet)...                                                         data_catalog.py:382
                    INFO     Saving data to 'y_test' (MemoryDataSet)...                                                          data_catalog.py:382
                    INFO     Completed 1 out of 3 tasks                                                                      sequential_runner.py:85
                    INFO     Loading data from 'X_train' (MemoryDataSet)...                                                      data_catalog.py:343
                    INFO     Loading data from 'X_test' (MemoryDataSet)...                                                       data_catalog.py:343
                    INFO     Loading data from 'y_train' (MemoryDataSet)...                                                      data_catalog.py:343
                    INFO     Running node: make_predictions: make_predictions([X_train,X_test,y_train]) -> [y_pred]                      node.py:329
...
```

You can see that the intermediate data is shared with memory. When kedro is deploy to a cluster, e.g.,
to [argo workflow][3], the `MemoryDataSet` is not applicable anymore and you will need to setup the
AWS S3 or Minio service and sharing those intermediate data as CSV files.

```yaml
X_train:
  type: pandas.CSVDataSet
  filepath: s3://testing/data/02_intermediate/X_train.csv
  credentials: minio

X_test:
  type: pandas.CSVDataSet
  filepath: s3://testing/data/02_intermediate/X_test.csv
  credentials: minio

y_train:
  type: pandas.CSVDataSet
  filepath: s3://testing/data/02_intermediate/y_train.csv
  credentials: minio
```

It might be inefficient for pickling pandas dataframes when data become larger. With the kedro
vineyard plugin, you can run the pipeline with vineyard as the intermediate data medium by

```bash
$ kedro run --runner vineyard.contrib.kedro.runner.SequentialRunner
[05/25/23 11:45:34] INFO     Kedro project iris                                                                                       session.py:355
                    INFO     Loading data from 'example_iris_data' (CSVDataSet)...                                               data_catalog.py:343
                    INFO     Loading data from 'parameters' (MemoryDataSet)...                                                   data_catalog.py:343
                    INFO     Running node: split: split_data([example_iris_data,parameters]) -> [X_train,X_test,y_train,y_test]          node.py:329
                    INFO     Saving data to 'X_train' (VineyardDataSet)...                                                       data_catalog.py:382
                    INFO     Saving data to 'X_test' (VineyardDataSet)...                                                        data_catalog.py:382
                    INFO     Saving data to 'y_train' (VineyardDataSet)...                                                       data_catalog.py:382
                    INFO     Saving data to 'y_test' (VineyardDataSet)...                                                        data_catalog.py:382
                    INFO     Loading data from 'X_train' (VineyardDataSet)...                                                    data_catalog.py:343
                    INFO     Loading data from 'X_test' (VineyardDataSet)...                                                     data_catalog.py:343
                    INFO     Loading data from 'y_train' (VineyardDataSet)...                                                    data_catalog.py:343
                    INFO     Running node: make_predictions: make_predictions([X_train,X_test,y_train]) -> [y_pred]                      node.py:329
...
```

Without any modification to your pipeline code, you can see that the intermediate data is shared
with vineyard using the `VineyardDataSet` and no longer suffers from the overhead of (de)serialization
and the I/O cost between external AWS S3 or Minio services.

Like `kedro catalog create`, the Kedro vineyard plugin provides a command-line interface to generate
the catalog configuration for given pipeline, which will rewrite the unspecified intermediate data
to `VineyardDataSet`, e.g.,

```bash
$ kedro vineyard catalog create -p __default__
```

You will get

```yaml
X_test:
  ds_name: X_test
  type: vineyard.contrib.kedro.io.dataset.VineyardDataSet
X_train:
  ds_name: X_train
  type: vineyard.contrib.kedro.io.dataset.VineyardDataSet
y_pred:
  ds_name: y_pred
  type: vineyard.contrib.kedro.io.dataset.VineyardDataSet
y_test:
  ds_name: y_test
  type: vineyard.contrib.kedro.io.dataset.VineyardDataSet
y_train:
  ds_name: y_train
  type: vineyard.contrib.kedro.io.dataset.VineyardDataSet
```

Deploy to Kubernetes
--------------------

When the pipeline scales to Kubernetes, the interaction with the Kedro vineyard plugin is
still simple and non-intrusive. The plugin provides tools to prepare the docker image and
generate Argo workflow specification file for the Kedro pipeline. Next, we'll demonstrate
how to deploy pipelines to Kubernetes while leverage Vineyard for efficient intermediate
sharing between tasks step-by-step.

1. Prepare the vineyard cluster (see also [Deploy on Kubernetes][5]):

   ```bash
   # export your kubeconfig path here
   $ export KUBECONFIG=/path/to/your/kubeconfig

   # install the vineyard operator
   $ go run k8s/cmd/main.go deploy vineyard-cluster --create-namespace
   ```

2. Install the argo server:

   ```bash
   # install the argo server
   $ kubectl create namespace argo
   $ kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml
   ```

3. Generate the iris demo project from the official template:

   ```bash
   $ kedro new --starter=pandas-iris
   ```

4. Build the Docker image for this iris demo project:

   ```bash
   # walk to the iris demo root directory
   $ cd iris
   $ kedro vineyard docker build
   ```

   A Docker image named `iris` will be built successfully. The docker image
   need to be pushed to your image registry, or loaded to the kind/minikube cluster, to be
   available in Kubernetes.

   ```bash
   $ docker images | grep iris
   iris   latest   3c92da8241c6   About a minute ago   690MB
   ```

5. Next, generate the Argo workflow YAML file from the iris demo project:

   ```bash
   $ kedro vineyard argo generate -i iris

   # check the generated Argo workflow YAML file, you can see the Argo workflow YAML file named `iris.yaml`
   # is generated successfully.
   $ ls -l argo-iris.yml
   -rw-rw-r-- 1 root root 3685 Jun 12 23:55 argo-iris.yml
   ```

6. Finally, submit the Argo workflow to Kubernetes:

   ```bash
   $ argo submit -n argo argo-iris.yml
   ```

   You can interact with the Argo workflow using the `argo` command-line tool, e.g.,

   ```bash
   $ argo list workflows -n argo
   NAME         STATUS      AGE   DURATION   PRIORITY   MESSAGE
   iris-sg6qf   Succeeded   18m   30s        0
   ```

We have prepared a benchmark to evaluate the performance gain brought by vineyard for data
sharing when data scales, for more details, please refer to [this report][4].

[1]: https://v6d.io/notes/getting-started.html#starting-vineyard-server
[2]: https://docs.kedro.org/en/stable/get_started/new_project.html
[3]: https://docs.kedro.org/en/stable/deployment/argo.html#how-to-run-your-kedro-pipeline-using-argo-workflows
[4]: https://v6d.io/tutorials/data-processing/accelerate-data-sharing-in-kedro.html
[5]: https://v6d.io/notes/cloud-native/deploy-kubernetes.html
