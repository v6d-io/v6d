## Use vineyard to accelerate kubeflow pipelines

Vineyard can accelerate data sharing by utilizing shared memory compared to existing methods such as local files or S3 services. In this doc, we will show you how to use vineyard to accelerate an existing kubeflow pipeline.


### Prerequisites

- Install the argo CLI tool via the [official guide](https://github.com/argoproj/argo-workflows/releases/).


### Overview of the pipeline

The pipeline we use is a simple pipeline that trains a linear regression model on the dummy Boston Housing Dataset. It contains three steps: preprocess, train, and test.


### Run the pipeline

Assume we have installed [kubeflow](https://www.kubeflow.org/docs/components/pipelines/v1/installation/standalone-deployment/#deploying-kubeflow-pipelines) and [vineyard](https://v6d.io/notes/cloud-native/deploy-kubernetes.html#quick-start) in the kubernetes cluster. We can use the following steps to run the pipeline:

First, we need to prepare the dataset by running the following command:

```bash
$ kubectl apply -f prepare_dataset.yaml
```

The dataset will be stored in the host path. Also, you may need to wait for a while for the dataset to be generated and you can use the following command to check the status:

```bash
$ kubectl logs -l app=prepare-data -n kubeflow | grep "preparing data time" >/dev/null && echo "dataset ready" || echo "dataset unready"
```

After that, you can run the pipeline via the following command:

```bash
$ argo submit --watch pipeline-with-vineyard.yaml -p data_multiplier=4000 -p registry="ghcr.io/v6d-io/v6d/kubeflow-example" -n kubeflow
```


### Modifications to use vineyard

Compared to the original kubeflow pipeline, we could use the following command to check the differences:

```bash
$ git diff --no-index --unified=40 pipeline.py pipeline-with-vineyard.py
```

The main modifications are:
- Add a new volume to the pipeline. This volume is used to connect to the vineyard cluster via the IPC socket file in
the host path.
- Add the scheduler annotations and labels to the pipeline. This is used to schedule the pipeline to the node that has vineyardd running.

Also, you can check the modifications of the source code as 
follows.

- [Save data in the preparation step](https://github.com/v6d-io/v6d/blob/main/k8s/examples/vineyard-kubeflow/preprocess/preprocess.py#L62-L72).
- [Load data in the training step](https://github.com/v6d-io/v6d/blob/main/k8s/examples/vineyard-kubeflow/train/train.py#L15-L24).
- [load data in the testing step](https://github.com/v6d-io/v6d/blob/main/k8s/examples/vineyard-kubeflow/test/test.py#L14-L20).

The main modification is to use vineyard to load and save data
rather than using local files.
