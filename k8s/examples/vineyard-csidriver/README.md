# Kubeflow Example with Vineyard CSI Driver

## Prerequisites

If you don't have a kubernetes cluster by hand, you can use [kind](https://kind.sigs.k8s.io/) to create a kubernetes cluster:

```bash
$ cat <<EOF | kind create cluster --config=-
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
    - role: control-plane
      image: kindest/node:v1.25.11
    - role: worker
      image: kindest/node:v1.25.11
    - role: worker
      image: kindest/node:v1.25.11
    - role: worker
      image: kindest/node:v1.25.11
EOF
```

Install the [Vineyardctl](https://v6d.io/notes/developers/build-from-source.html#install-vineyardctl) by following the official guide.

## Build CSI Driver image

The csi driver is built in the vineyard csidriver image. You can build the image by running:

```bash
$ cd k8s && make csidriver
```

Load the image to the kind cluster:

```bash
$ kind load docker-image vineyardcloudnative/vineyard-csi-driver:latest
```

## Deploy vineyard cluster

Deploy vineyard cluster to the kind cluster:

```bash
$ vineyardctl deploy vineyard-cluster --create-namespace
```

## Deploy vineyard CSI Driver

Deploy vineyard CSI Driver to the kind cluster:

```bash
$ cat <<EOF | kubectl apply -f -
apiVersion: k8s.v6d.io/v1alpha1
kind: CSIDriver
metadata:
  name: csidriver-sample
spec:
  clusters:
  - namespace: vineyard-system
    name: vineyardd-sample
EOF
```

## Install the argo workflow

```bash
$ kubectl create ns argo
$ kubectl apply -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml
```

## Check Argo Workflow

The Argo workflow `pipeline.yaml` and `workflow-with-vineyard.yaml` are in the `k8s/examples/vineyard-csidriver` directory. They are built from the 
Kubeflow API, but for the convenience of benchmarking, we have modified the original generated argo workflow.

## Run the kubeflow example

Build the docker image of the kubeflow example and load them to the kind cluster:

```bash
$ cd k8s/examples/vineyard-csidriver
$ make docker-build && make load-images
```

To simulate the data loading/saving of the actual pipeline, we use the nfs volume to store the data. The nfs volume is mounted to the `/mnt/data` directory of the kind cluster. Then apply the data volume as follows:

```bash
$ kubectl apply -f prepare-data.yaml
```

Deploy the rbac for the kubeflow example:

```bash
$ kubectl apply -f rbac.yaml
```

Submit the kubeflow example without vineyard to the argo server:

```bash
$ for data_multiplier in 3000 4000 5000; do \
     argo submit --watch pipeline.yaml -p data_multiplier=${data_multiplier}; \
done
```

Clear the previous resources:

```bash
$ argo delete --all
```

Submit the kubeflow example with vineyard to the argo server:

```bash
$ for data_multiplier in 3000 4000 5000; do \
     argo submit --watch pipeline-with-vineyard.yaml -p data_multiplier=${data_multiplier}; \
done
```

## Clean up

Delete the rbac for the kubeflow example:

```bash
$ kubectl delete -f rbac.yaml
```

Delete all argo workflow

```bash
$ argo delete --all
```

Delete the argo server:

```bash
$ kubectl delete ns argo
```

Delete the csi driver:

```bash
$ kubectl delete csidrivers.k8s.v6d.io csidriver-sample
```

Delete the vineyard cluster:

```bash
$ vineyardctl delete vineyard-cluster
```

Delete the data volume:

```bash
$ kubectl delete -f prepare-data.yaml
```

## Result Analysis

The time of argo workflow execution of the pipeline is as follows:

### Argo workflow duration

| data_multiplier | without vineyard | with vineyard |
| --------------- | ---------------- | ------------- |
| 3000(8.5G)      |    186s          |      169s     |
| 4000(12G)       |    250s          |      203s     |
| 5000(15G)       |    332s          |      286s     |


Actually, the cost time of argo workflow is affected by lots of factors, such as the network, the cpu and memory of the cluster, the data volume, etc. So the time of argo workflow execution of the pipeline is not stable. But we can still find that the time of argo workflow execution of the pipeline with vineyard is shorter than that without vineyard.

Also, we record the whole execution time via logs. The result is as follows:

### Actual execution time

| data_multiplier | without vineyard | with vineyard |
| --------------- | ---------------- | ------------- |
| 3000            |    139.3s        |      92.3s    |
| 4000            |    204.3s        |      131.1s   |
| 5000            |    289.3s        |      209.7s   |

According to the above results, we can find that the time of actual execution of the pipeline with vineyard is shorter than that without vineyard. To be specific, we record the write/read time of the following steps:

### Write time

| data_multiplier | without vineyard | with vineyard |
| --------------- | ---------------- | ------------- |
| 3000            |    21s           |      5.4s     |
| 4000            |    26s           |      7s       |
| 5000            |    32.2s         |      9.4s     |

From the above results, we can find that the time of write of the pipeline with vineyard is nearly 4 times shorter than that without vineyard. The reason is that the data is stored in the vineyard cluster, so it's actually a memory copy operation, which is faster than the write operation of the nfs volume.


### Read time(Delete the time of init data loading)

| data_multiplier | without vineyard | with vineyard |
| --------------- | ---------------- | ------------- |
| 3000            |    36.7s         |      0.02s    |
| 4000            |    45.7s         |      0.02s    |
| 5000            |    128.6s        |      0.04s    |

Based on the above results, we can find that the read time of vineyard is nearly a constant, which is not affected by the data scale. The reason is that the data is stored in the shared memory of vineyard cluster, so it's actually a pointer copy operation.

As a result, we can find that with vineyard, the argo workflow duration of the pipeline is reduced by 10%~20% and the actual execution time of the pipeline is reduced by about 30%.