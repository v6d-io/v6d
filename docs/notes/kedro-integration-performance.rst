.. _kedro-integration-performance:

Kedro Integration Performance Report
====================================

This is a performance report of kedro integration, here we will compare three
different data catalog of kedro benchmark project: vineyard(v0.15.3), AWS S3 and MinIO S3(the latest one).

Create a kubernetes cluster
---------------------------

If you don't have a kubernetes on hand, you can use the `kind v0.20.0 <https://kind.sigs.k8s.io/>`_
to create a kubernetes cluster with 4 nodes(including a master node) as follows:

.. code:: bash

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
    NAME                                READY   STATUS    RESTARTS   AGE
    argo-server-7698c96655-jg2ds        1/1     Running   0          11s
    workflow-controller-b888f4458-x4qf2 1/1     Running   0          11s


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

1. Deploy the minio resources.

.. code:: bash

    $ kubectl apply -f python/vineyard/contrib/kedro/benchmark/minio-s3/minio-dev.yaml

2. The default access key and secret key of the minio cluster are `minioadmin` and `minioadmin`.

3. Install the secret of the MinIO cluster.

.. code:: bash

    $ cat <<EOF | kubectl apply -n minio-dev -f -
    apiVersion: v1
    kind: Secret
    metadata:
      name: my-minio-cred
    type: Opaque
    data:
      accessKey: <Your Access Key> and Base64 encoded
      secretKey: <Your Secret Key> and Base64 encoded
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

    # Get the endpoint of minio service
    $ minioUrl=$(kubectl get endpoints -n minio-dev -o jsonpath='{.items[*].subsets[*].addresses[*].ip}'):9000

    # Replace with actual minio url
    $ sed -i "s/{{MINIO}}/${minioUrl}/g" ./minio-default.yaml

    # Apply to configmap in the argo namespace
    $ kubectl -n argo patch configmap/workflow-controller-configmap --patch "$(cat ./minio-default.yaml)"

5. Forward minio-artifacts service.

.. code:: bash

    $ kubectl port-forward service/minio -n minio-dev 9000:9000

6. Download the minio client and install it.

.. code:: bash

    $ wget https://dl.min.io/client/mc/release/linux-amd64/mc
    $ chmod +x mc
    $ sudo mv mc /usr/local/bin

7. Configure the minio client.

.. code:: bash

    $ mc alias set minio http://localhost:9000
    Enter Access Key: <Your Access Key>
    Enter Secret Key: <Your Secret Key>

1. Create a bucket named `minio-s3-benchmark-bucket` on the MinIO cluster.

.. code:: bash

    $ mc mb minio/minio-s3-benchmark-bucket
    Bucket created successfully `minio/minio-s3-benchmark-bucket`.


Prepare the kedro benchmark project
-----------------------------------

1. Go to the kedro benchmark project under vineyard root directory.

.. code:: bash

    $ cd python/vineyard/contrib/kedro/benchmark

2. Fulfill the credentials configurations of AWS S3.

.. code:: bash

    $ cd aws-s3
    $ cat conf/local/credentials.yml
    benchmark_aws_s3:
        client_kwargs:
            aws_access_key_id: Your AWS Access Key ID
            aws_secret_access_key: Your AWS Secret Access Key
            region_name: Your AWS Region Name

2. Build the docker images of the kedro project for vineyard benchmark.

.. code:: bash

    $ pushd vineyard
    # build the docker images
    $ make
    # check the docker images
    $ docker images | grep vineyard-benchmark
    vineyard-benchmark-with-500m-data   latest  982c6a376597   About a minute ago   1.66GB
    vineyard-benchmark-with-100m-data   latest  e58ca1cada98   About a minute ago   1.25GB
    vineyard-benchmark-with-10m-data    latest  f7c618b48913   About a minute ago   1.16GB
    vineyard-benchmark-with-1m-data     latest  8f9e74ff5116   About a minute ago   1.15GB
    $ popd

3. Build the docker images of the kedro project for aws s3 benchmark.

.. code:: bash

    $ pushd aws-s3
    # build the docker images
    $ make
    # check the docker images
    $ docker images | grep aws-s3-benchmark
    aws-s3-benchmark-with-500m-data latest  877d8fc1ef78   3 minutes ago   1.42GB
    aws-s3-benchmark-with-100m-data latest  b8e15edda5cd   3 minutes ago   1.01GB
    aws-s3-benchmark-with-10m-data  latest  c1a58ddb2888   3 minutes ago   915MB
    aws-s3-benchmark-with-1m-data   latest  9f27ac5ce9dd   3 minutes ago   907MB
    $ popd

4. Build the docker images of the kedro project for minio s3 benchmark.

.. code:: bash

    $ pushd minio-s3

    # build the docker images
    $ make

    # check the docker images
    $ docker images | grep minio-s3-benchmark
    minio-s3-benchmark-with-500m-data   latest  1c75300390cf   8 seconds ago    1.41GB
    minio-s3-benchmark-with-100m-data   latest  f4aa093ddf36   11 seconds ago   1.01GB
    minio-s3-benchmark-with-10m-data    latest  8b068600e368   12 seconds ago   913MB
    minio-s3-benchmark-with-1m-data     latest  b3eaf0a5898c   13 seconds ago   904MB

    $ popd

5. Load the above images to the kind cluster.

.. code:: bash

    # load the vineyard benchmark images
    $ kind load docker-image vineyard-benchmark-with-1m-data && \
        kind load docker-image vineyard-benchmark-with-10m-data && \
        kind load docker-image vineyard-benchmark-with-100m-data && \
        kind load docker-image vineyard-benchmark-with-500m-data
    # load the aws s3 benchmark images
    $ kind load docker-image aws-s3-benchmark-with-1m-data && \
        kind load docker-image aws-s3-benchmark-with-10m-data && \
        kind load docker-image aws-s3-benchmark-with-100m-data && \
        kind load docker-image aws-s3-benchmark-with-500m-data
    # load the minio s3 benchmark images
    $ kind load docker-image minio-s3-benchmark-with-1m-data && \
        kind load docker-image minio-s3-benchmark-with-10m-data && \
        kind load docker-image minio-s3-benchmark-with-100m-data && \
        kind load docker-image minio-s3-benchmark-with-500m-data


Submit the benchmark workflow
-----------------------------

1. Submit the vineyard benchmark workflow.

.. code:: bash

    $ pushd vineyard
    # 1M data
    $ sed -i "s/vineyard-benchmark/vineyard-benchmark-with-1m-data/g" argo-vineyard-benchmark.yml && \
        argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 10M data
    $ sed -i "s/vineyard-benchmark-with-1m-data/vineyard-benchmark-with-10m-data/g" argo-vineyard-benchmark.yml && \
        argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 100M data
    $ sed -i "s/vineyard-benchmark-with-10m-data/vineyard-benchmark-with-100m-data/g" argo-vineyard-benchmark.yml && \
        argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    # 500M data
    $ sed -i "s/vineyard-benchmark-with-100m-data/vineyard-benchmark-with-500m-data/g" argo-vineyard-benchmark.yml && \
        argo submit -n vineyard-system --watch argo-vineyard-benchmark.yml
    $ popd

2. Submit the aws s3 benchmark workflow.

.. code:: bash

    $ pushd aws-s3
    # 1M data
    $ sed -i "s/aws-s3-benchmark/aws-s3-benchmark-with-1m-data/g" argo-aws-s3-benchmark.yml && \
        argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 10M data
    $ sed -i "s/aws-s3-benchmark-with-1m-data/aws-s3-benchmark-with-10m-data/g" argo-aws-s3-benchmark.yml && \
        argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 100M data
    $ sed -i "s/aws-s3-benchmark-with-10m-data/aws-s3-benchmark-with-100m-data/g" argo-aws-s3-benchmark.yml && \
        argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    # 500M data
    $ sed -i "s/aws-s3-benchmark-with-100m-data/aws-s3-benchmark-with-500m-data/g" argo-aws-s3-benchmark.yml && \
        argo submit -n s3 --watch argo-aws-s3-benchmark.yml
    $ popd

3. Submit the minio s3 benchmark workflow.

.. code:: bash

    $ pushd minio-s3
    # 1M data
    $ sed -i "s/minio-s3-benchmark/minio-s3-benchmark-with-1m-data/g" argo-minio-s3-benchmark.yml && \
        argo submit -n minio-dev --watch argo-minio-s3-benchmark.yml
    # 10M data
    $ sed -i "s/minio-s3-benchmark-with-1m-data/minio-s3-benchmark-with-10m-data/g" argo-minio-s3-benchmark.yml && \
        argo submit -n minio-dev --watch argo-minio-s3-benchmark.yml
    # 100M data
    $ sed -i "s/minio-s3-benchmark-with-10m-data/minio-s3-benchmark-with-100m-data/g" argo-minio-s3-benchmark.yml && \
        argo submit -n minio-dev --watch argo-minio-s3-benchmark.yml
    # 500M data
    $ sed -i "s/minio-s3-benchmark-with-100m-data/minio-s3-benchmark-with-500m-data/g" argo-minio-s3-benchmark.yml && \
        argo submit -n minio-dev --watch argo-minio-s3-benchmark.yml

4. Record the time of each workflow.


Summary
-------

After running the benchmark, we can get the following results:

The data size is the size of input file, and the time is
the completion time of the argo workflow.

| Data Size | Vineyard |  AWS S3  | MinIO S3 |
| --------- | -------- | -------- | -------- |
| 1M        | 30s      | 50s      | 30s      |
| 10M       | 30s      | 63s      | 30s      |
| 100M      | 60s      | 144s     | 64s      |
| 500M      | 91s      | 457s     | 177s     |
