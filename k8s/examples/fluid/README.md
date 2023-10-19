## Fluid integration

If you are using [Fluid](https://fluidframework.com/) in your application, now it's a chance to cache your **Python Object** using **Vineyard** based on **Fluid**.

### Prerequisites 

- A kubernetes cluster with version >= 1.25.10. If you don't have one by hand, you can refer to the guide [Initialize Kubernetes Cluster](https://v6d.io/tutorials/kubernetes/using-vineyard-operator.html#step-0-optional-initialize-kubernetes-cluster) to create one.
- Install the [Vineyardctl](https://v6d.io/notes/developers/build-from-source.html#install-vineyardctl) by following the official guide.

### Install the argo server on Kubernetes

1. Install the argo server on Kubernetes:

```bash
$ kubectl create namespace argo
$ kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml
```

2. Check the status of the argo server:

```bash
$ kubectl get pod -n argo
NAME                                  READY   STATUS    RESTARTS   AGE
argo-server-7698c96655-xsd5k          1/1     Running   0          10m
workflow-controller-b888f4458-ts58f   1/1     Running   0          10m
```

### Submit the job wihout Vineyard

1. Submit the workflow:

```bash
$ cd k8s/examples/fluid
$ argo submit --watch argo_workflow.yaml
```

### Submit the job with Vineyard


1. Install the vineyard deployment:

```bash
$ vineyardctl deploy vineyard-deployment
```

2. Install the Fluid:

```bash
$ kubectl create ns fluid-system
$ helm repo add fluid https://fluid-cloudnative.github.io/charts
$ helm repo update
$ helm install fluid fluid/fluid
``` 

3. Build the `configure-vineyard-socket` image and load to the cluster.

```bash
$ cd k8s/examples/fluid && make build-image && kind load docker-image configure-vineyard-socket
```

4. Install the Vineyard profile:

```bash
$ kubectl apply -f vineyard_profile.yaml
```

5. Install the Vineyard Dataset:

```bash
$ kubectl apply -f dataset.yaml
```

6. Check the dataset status and make sure the dataset is in `Bound` status as follows.

```bash
$ kubectl get dataset -A
NAMESPACE   NAME       UFS TOTAL SIZE   CACHED   CACHE CAPACITY   CACHED PERCENTAGE   PHASE   AGE
default     vineyard   [Calculating]    N/A      N/A              N/A                 Bound   105s
```

After that, the vineyard pv and pvc will be created automatically as follows. 

```bash
$ kubectl get pvc   
NAME       STATUS   VOLUME             CAPACITY   ACCESS MODES   STORAGECLASS   AGE
vineyard   Bound    default-vineyard   100Pi      RWX            fluid          87s
$ kubectl get pv
NAME               CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM              STORAGECLASS   REASON   AGE
default-vineyard   100Pi      RWX            Retain           Bound    default/vineyard   fluid                   2m43s
```

7. Submit the workflow with Vineyard:

```bash
$ argo submit --watch argo_workflow_with_vineyard.yaml
```

### result

The execution time of the workflow without Vineyard is 41s.

