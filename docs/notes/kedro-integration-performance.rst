.. _kedro-integration-performance:

Kedro Integration Performance
=============================

This is a performance report of kedro integration, here we will compare the Vineyard as
the data catalog of kedro with the s3 service built with Minio on the Kubernetes cluster.

Create a kubernetes cluster
---------------------------

Use the [kind](https://kind.sigs.k8s.io/) to create a kubernetes cluster with 4
nodes(including a master node) as follows:

.. code:: bash

    $ cat <<EOF | kind create cluster --config=-
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
    - role: control-plane
    image: kindest/node:v1.24.0
    - role: worker
    image: kindest/node:v1.24.0
    - role: worker
    image: kindest/node:v1.24.0
    - role: worker
    image: kindest/node:v1.24.0
    EOF

Install MinIO
-------------

1. Create the minio namespace as follows:

.. code:: bash

    $ kubectl create namespace minio

2. Install the MinIO cluster via helm chart.

.. code:: bash

    helm install --namespace=minio minio-artifacts stable/minio --set service.type=LoadBalancer --set fullnameOverride=argo-artifacts --set b2gateway.enabled=true

3. Install the secret of the MinIO cluster.

.. code:: bash

    $ cat <<EOF | kubectl apply -n minio -f -
    apiVersion: v1
    kind: Secret
    metadata:
      name: my-minio-cred
    type: Opaque
    data:
      accessKey: QUtJQUlPU0ZPRE5ON0VYQU1QTEU= # AKIAIOSFODNN7EXAMPLE
      secretKey: d0phbHJYVXRuRkVNSS9LN01ERU5HL2JQeFJmaUNZRVhBTVBMRUtFWQ== #wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    EOF

4. Set the configurations of MinIO clusters.

.. code:: bash

    $ cat << EOF > minio-default.yaml
    data:
    artifactRepository: |
      s3:
        bucket: my-bucket
        keyFormat: prefix/in/bucket
        endpoint: {{MINIO}}
        insecure: true
        accessKeySecret:
          name: my-minio-cred
          key: accessKey
        secretKeySecret:
          name: my-minio-cred
          key: secretKey
        useSDKCreds: false
    EOF
    # Get the actual MinIO service address.
    $ minioUrl=$(kubectl get service minio-artifacts -n argo -o jsonpath='{.spec.clusterIP}:{.spec.ports[0].nodePort}')
    
    # Replace with actual minio url
    $ sed -i "s/{{MINIO}}/${minioUrl}/g" ./minio-default.yaml

    # Apply to k8s in the argo namespace
    $ kubectl -n argo patch configmap/workflow-controller-configmap --patch "$(cat ./minio-default.yaml)"

5. Forward minio-artifacts service.

.. code:: bash

    $ kubectl port-forward service/minio-artifacts -n minio 9000:9000

6. Open the website `http://127.0.0.1:9000` and login with the following credentials.

.. code:: bash
    
    # Access Key
    AKIAIOSFODNN7EXAMPLE

    # Secret Key
    wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

7. Create a bucket named `my-bucket` on the MinIO cluster.


Install Vineyard Operator
-------------------------

1. Deploy the cert-manager.

.. code:: bash

    $ go run k8s/cmd/main.go deploy cert-manager

2. Deploy the vineyard operator.

.. code:: bash

    $ go run k8s/cmd/main.go deploy vineyard-operator

3. Deploy the vineyard cluster.

As the memory of minio cluster is 4G, we set the memory of vineyard cluster to 4G as well.

.. code:: bash

    $ go run k8s/cmd/main.go deploy vineyardd --vineyardd.memory=4Gi --vineyardd.size=4Gi

Install the argo server
-----------------------

1. Create the argo namespace.

.. code:: bash

    $ kubectl create namespace argo

2. Install the argo server.

.. code:: bash

    $ kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml

3. Check the argo server.

.. code:: bash

    $ kubectl get pod -n argo
    NAME                                READY   STATUS    RESTARTS   AGE                                                          │
    argo-server-7698c96655-jg2ds        1/1     Running   0          11s                                                          
    workflow-controller-b888f4458-x4qf2 1/1     Running   0          11s


Prepare the kedro project
-------------------------

1. Download the kedro project.

.. code:: bash

    $ git clone  https://github.com/dashanji/kedro-benchmark-project.git

2. Build the docker images of the kedro project for minio benchmark.

.. code:: bash

    $ cd minio-benchmark && make
    # check the docker images
    $ docker images | grep minio-benchmark
    minio-benchmark-with-500m-data   latest    c905cbd720d6   About a minute ago   1.54GB
    minio-benchmark-with-100m-data   latest    74bec4b9f89a   2 minutes ago        1.14GB
    minio-benchmark-with-10m-data    latest    85a52bd54dc1   2 minutes ago        1.05GB
    minio-benchmark-with-1m-data     latest    7b9a24f77987   2 minutes ago        1.04GB

3. Build the docker images of the kedro project for vineyard benchmark.

.. code:: bash

    $ cd vineyard-benchmark && make
    # check the docker images
    $ docker images | grep vineyard-benchmark
    vineyard-benchmark-with-500m-data      latest    06e25a7f1257   5 minutes ago    2.05GB
    vineyard-benchmark-with-100m-data      latest    10ee73a42184   5 minutes ago    1.64GB
    vineyard-benchmark-with-10m-data       latest    8b62806d8e96   5 minutes ago    1.55GB
    vineyard-benchmark-with-1m-data        latest    8e871960be3f   5 minutes ago    1.54GB

4. Load the minio benchmark images to the kind cluster.

.. code:: bash

    $ kind load docker-image minio-benchmark-with-1m-data && kind load docker-image minio-benchmark-with-10m-data && kind load docker-image minio-benchmark-with-100m-data && kind load docker-image minio-benchmark-with-500m-data
    
5. Load the vineyard benchmark images to the kind cluster.

.. code:: bash

    $ kind load docker-image vineyard-benchmark-with-1m-data && kind load docker-image vineyard-benchmark-with-10m-data && kind load docker-image vineyard-benchmark-with-100m-data && kind load docker-image vineyard-benchmark-with-500m-data

Submit the benchmark workflow
-----------------------------

1. Submit the minio benchmark workflow.

.. code:: bash

    # 1M data
    $ sed -i "s/minio-benchmark/minio-benchmark-with-1m-data/g" argo-minio-benchmark.yml && argo submit -n argo --watch argo-minio-benchmark.yml
    # 10M data
    $ sed -i "s/minio-benchmark-with-1m-data/minio-benchmark-with-10m-data/g" argo-minio-benchmark.yml && argo submit -n argo --watch argo-minio-benchmark.yml
    # 100M data
    $ sed -i "s/minio-benchmark-with-10m-data/minio-benchmark-with-100m-data/g" argo-minio-benchmark.yml && argo submit -n argo --watch argo-minio-benchmark.yml
    # 500M data
    $ sed -i "s/minio-benchmark-with-100m-data/minio-benchmark-with-500m-data/g" argo-minio-benchmark.yml && argo submit -n argo --watch argo-minio-benchmark.yml

2. Submit the vineyard benchmark workflow.

.. code:: bash

    # 1M data
    $ sed -i "s/vineyard-benchmark/vineyard-benchmark-with-1m-data/g" argo-vineyard-benchmark.yml && argo submit -n argo --watch argo-vineyard-benchmark.yml
    # 10M data
    $ sed -i "s/vineyard-benchmark-with-1m-data/vineyard-benchmark-with-10m-data/g" argo-vineyard-benchmark.yml && argo submit -n argo --watch argo-vineyard-benchmark.yml
    # 100M data
    $ sed -i "s/vineyard-benchmark-with-10m-data/vineyard-benchmark-with-100m-data/g" argo-vineyard-benchmark.yml && argo submit -n argo --watch argo-vineyard-benchmark.yml
    # 500M data
    $ sed -i "s/vineyard-benchmark-with-100m-data/vineyard-benchmark-with-500m-data/g" argo-vineyard-benchmark.yml && argo submit -n argo --watch argo-vineyard-benchmark.yml

3. Record the time of each workflow.

Summary
-------

After running the benchmark, we can get the following results:

The data size is the size of input file, and the time is 
the completion time of the argo workflow.

| Data Size | Vineyard | MinIO |
| --------- | -------- | ----- |
| 1M        | 30s      | 30s   |
| 10M       | 30s      | 40s   |
| 100M      | 61s      | 74s   |
| 500M      | 114s     | 225s  |