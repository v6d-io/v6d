.. _kedro-integration-performance:

Kedro Integration Performance Report
====================================

This is a performance report of kedro integration, here we will compare three
different data catalog of kedro benchmark project: vineyard, AWS S3 and MinIO S3.


Create a kubernetes cluster
---------------------------

If you don't have a kubernetes on hand, you can use the [kind](https://kind.sigs.k8s.io/)
to create a kubernetes cluster with 4 nodes(including a master node) as follows:

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


Install Vineyard Operator
-------------------------

1. Deploy the vineyard operator.

.. code:: bash

    $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
    $ helm repo update
    $ helm install vineyard-operator vineyard/vineyard-operator \
      --namespace vineyard-system \
      --create-namespace

2. Deploy the vineyard cluster.

To handle the large data, we set the memory of vineyard cluster to 4G as well.

.. code:: bash

    $ python3 -c "import vineyard; vineyard.deploy.vineyardctl.deploy.vineyardd(vineyardd_memory='4Gi', vineyardd_size='4Gi')"


Prepare AWS S3
--------------

1. Create the s3 namespace.

.. code:: bash

    $ kubectl create namespace s3

2. Deploy the secret of AWS S3 credentials.

.. code:: bash

    $ cat <<EOF | kubectl apply -n s3 -f -
    # secret.yml
    apiVersion: v1
    kind: Secret
    metadata:
      name: aws-secrets
    data:
      access_key_id: Your AWS Access Key ID and Base64 encoded
      secret_access_key: Your AWS Secret Access Key and Base64 encoded
    type: Opaque
    EOF

3. Create a bucket named `aws-s3-benchmark-bucket` on the AWS S3.


Install MinIO S3
----------------

1. Create the minio namespace as follows:

.. code:: bash

    $ kubectl create namespace minio

2. Install the MinIO cluster via helm chart.

.. code:: bash

    $ helm repo add stable https://charts.helm.sh/stable
    $ helm repo update
    $ helm install --namespace=minio minio-artifacts stable/minio --set service.type=LoadBalancer --set fullnameOverride=minio-artifacts

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
          bucket: minio-s3-benchmark-bucket
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
    $ minioUrl=$(kubectl get service minio-artifacts -n minio -o jsonpath='{.spec.clusterIP}:{.spec.ports[0].nodePort}')
    
    # Replace with actual minio url
    $ sed -i "s/{{MINIO}}/${minioUrl}/g" ./minio-default.yaml
    # Apply to configmap in the argo namespace
    $ kubectl -n argo patch configmap/workflow-controller-configmap --patch "$(cat ./minio-default.yaml)"

5. Forward minio-artifacts service.

.. code:: bash

    $ kubectl port-forward service/minio-artifacts -n minio 9000:9000

6. Download the minio client and install it.

.. code:: bash

    $ wget https://dl.min.io/client/mc/release/linux-amd64/mc
    $ chmod +x mc
    $ sudo mv mc /usr/local/bin

7. Configure the minio client.

.. code:: bash

    $ mc alias set minio http://localhost:9000
    Enter Access Key: AKIAIOSFODNN7EXAMPLE
    Enter Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

8. Create a bucket named `minio-s3-benchmark-bucket` on the MinIO cluster.

.. code:: bash
    
    $ mc mb minio/minio-s3-benchmark-bucket
    Bucket created successfully `minio/minio-s3-benchmark-bucket`.


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


Prepare the kedro benchmark project
-----------------------------------

1. Download the kedro project.

.. code:: bash

    $ git clone https://github.com/dashanji/kedro-benchmark-project.git

2. Fulfill the credentials configurations of AWS S3.

.. code:: bash

    $ cd aws-s3-benchmark
    $ cat conf/local/credentials.yml
    benchmark_aws_s3:
        client_kwargs:
            aws_access_key_id: Your AWS Access Key ID
            aws_secret_access_key: Your AWS Secret Access Key

2. Build the docker images of the kedro project for vineyard benchmark.

.. code:: bash

    $ pushd vineyard-benchmark
    # build the docker images
    $ make
    # check the docker images
    $ docker images | grep vineyard-benchmark
    vineyard-benchmark-with-500m-data       latest    0430517cd6c3   48 minutes ago       2.26GB
    vineyard-benchmark-with-100m-data       latest    21532a9514e7   48 minutes ago       1.86GB
    vineyard-benchmark-with-10m-data        latest    83672e4baec2   49 minutes ago       1.77GB
    vineyard-benchmark-with-1m-data         latest    4506d2cc264a   49 minutes ago       1.76GB
    $ popd

3. Build the docker images of the kedro project for aws s3 benchmark.

.. code:: bash

    $ pushd aws-s3-benchmark
    # build the docker images
    $ make
    # check the docker images
    $ docker images | grep aws-s3-benchmark
    aws-s3-benchmark-with-500m-data         latest    f888ebff69a9   48 seconds ago      2.01GB
    aws-s3-benchmark-with-100m-data         latest    744852f72352   2 minutes ago       1.61GB
    aws-s3-benchmark-with-10m-data          latest    0e5dde266d7a   3 minutes ago       1.52GB
    aws-s3-benchmark-with-1m-data           latest    a6813fce87f8   4 minutes ago       1.51GB
    $ popd

4. Build the docker images of the kedro project for minio s3 benchmark.

.. code:: bash

    $ pushd minio-s3-benchmark
    # build the docker images
    $ make
    # check the docker images
    $ docker images | grep minio-s3-benchmark
    minio-s3-benchmark-with-500m-data       latest    bcee3927f4c5   49 minutes ago       2.01GB
    minio-s3-benchmark-with-100m-data       latest    624237fdc2e4   50 minutes ago       1.61GB
    minio-s3-benchmark-with-10m-data        latest    398084760ac7   50 minutes ago       1.52GB
    minio-s3-benchmark-with-1m-data         latest    c37c31629a3d   50 minutes ago       1.51GB
    $ popd

5. Load the above images to the kind cluster.

.. code:: bash
    # load the vineyard benchmark images
    $ kind load docker-image vineyard-benchmark-with-1m-data && kind load docker-image vineyard-benchmark-with-10m-data && kind load docker-image vineyard-benchmark-with-100m-data && kind load docker-image vineyard-benchmark-with-500m-data
    # load the aws s3 benchmark images
    $ kind load docker-image aws-s3-benchmark-with-1m-data && kind load docker-image aws-s3-benchmark-with-10m-data && kind load docker-image aws-s3-benchmark-with-100m-data && kind load docker-image aws-s3-benchmark-with-500m-data
    # load the minio s3 benchmark images
    $ kind load docker-image minio-s3-benchmark-with-1m-data && kind load docker-image minio-s3-benchmark-with-10m-data && kind load docker-image minio-s3-benchmark-with-100m-data && kind load docker-image minio-s3-benchmark-with-500m-data


Submit the benchmark workflow
-----------------------------

1. Submit the vineyard benchmark workflow.

.. code:: bash

    $ pushd vineyard-benchmark
    # 1M data
    $ sed -i "s/vineyard-benchmark/vineyard-benchmark-with-1m-data/g" argo-vineyard-benchmark.yml && argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 10M data
    $ sed -i "s/vineyard-benchmark-with-1m-data/vineyard-benchmark-with-10m-data/g" argo-vineyard-benchmark.yml && argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 100M data
    $ sed -i "s/vineyard-benchmark-with-10m-data/vineyard-benchmark-with-100m-data/g" argo-vineyard-benchmark.yml && argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 500M data
    $ sed -i "s/vineyard-benchmark-with-100m-data/vineyard-benchmark-with-500m-data/g" argo-vineyard-benchmark.yml && argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    $ popd

2. Submit the aws s3 benchmark workflow.

.. code:: bash

    $ pushd aws-s3-benchmark
    # 1M data
    $ sed -i "s/aws-s3-benchmark/aws-s3-benchmark-with-1m-data/g" argo-aws-s3-benchmark.yml && argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 10M data
    $ sed -i "s/aws-s3-benchmark-with-1m-data/aws-s3-benchmark-with-10m-data/g" argo-aws-s3-benchmark.yml && argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 100M data
    $ sed -i "s/aws-s3-benchmark-with-10m-data/aws-s3-benchmark-with-100m-data/g" argo-aws-s3-benchmark.yml && argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 500M data
    $ sed -i "s/aws-s3-benchmark-with-100m-data/aws-s3-benchmark-with-500m-data/g" argo-aws-s3-benchmark.yml && argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    $ popd

3. Submit the minio s3 benchmark workflow.

.. code:: bash

    $ pushd minio-s3-benchmark
    # 1M data
    $ sed -i "s/minio-s3-benchmark/minio-s3-benchmark-with-1m-data/g" argo-minio-s3-benchmark.yml && argo submit -n minio --watch argo-minio-s3-benchmark.yml
    # 10M data
    $ sed -i "s/minio-s3-benchmark-with-1m-data/minio-s3-benchmark-with-10m-data/g" argo-minio-s3-benchmark.yml && argo submit -n minio --watch argo-minio-s3-benchmark.yml
    # 100M data
    $ sed -i "s/minio-s3-benchmark-with-10m-data/minio-s3-benchmark-with-100m-data/g" argo-minio-s3-benchmark.yml && argo submit -n minio --watch argo-minio-s3-benchmark.yml
    # 500M data
    $ sed -i "s/minio-s3-benchmark-with-100m-data/minio-s3-benchmark-with-500m-data/g" argo-minio-s3-benchmark.yml && argo submit -n minio --watch argo-minio-s3-benchmark.yml
    
4. Record the time of each workflow.


Summary
-------

After running the benchmark, we can get the following results:

The data size is the size of input file, and the time is 
the completion time of the argo workflow.

| Data Size | Vineyard |  AWS S3  | MinIO S3 | 
| --------- | -------- | -------- | -------- |
| 1M        | 30s      | 61s      | 31s      |
| 10M       | 30s      | 63s      | 31s      |
| 100M      | 60s      | 141s     | 63s      |
| 500M      | 108s     | 418s     | 178s     |