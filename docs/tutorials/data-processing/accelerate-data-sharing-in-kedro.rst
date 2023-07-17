.. _accelerate-data-sharing-in-kedro:

Accelerate Data Sharing in Kedro
================================

This is a tutorial that shows how Vineyard accelerate the intermediate data
sharing between tasks in Kedro pipelines using our
`vineyard-kedro <https://pypi.org/project/vineyard-kedro/>`_ plugin, when data
scales and the pipeline are deployed on Kubernetes.

Prepare the Kubernetes cluster
------------------------------

To deploy Kedro pipelines on Kubernetes, you must have a kubernetes cluster.

.. tip::

    If you already have a K8s cluster, just skip this section and continue
    on deploying.

We recommend `kind v0.20.0 <https://kind.sigs.k8s.io/>`_ to create a multi-node
Kubernetes cluster on your local machine as follows:

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


Deploy Argo Workflows
---------------------

Install the argo operator on Kubernetes:

.. code:: bash

    $ kubectl create namespace argo
    $ kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml

When the deployment becomes ready, you can see the following pods:

.. code:: bash

    $ kubectl get pod -n argo
    NAME                                READY   STATUS    RESTARTS   AGE
    argo-server-7698c96655-jg2ds        1/1     Running   0          11s
    workflow-controller-b888f4458-x4qf2 1/1     Running   0          11s

Deploy Vineyard
---------------

1. Install the vineyard operator:

   .. code:: bash

       $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
       $ helm repo update
       $ helm install vineyard-operator vineyard/vineyard-operator \
           --namespace vineyard-system \
           --create-namespace

2. Create a vineyard cluster:

   .. tip::

       To handle the large data, we set the memory of vineyard cluster to 4G.

   .. code:: bash

       $ python3 -m vineyard.ctl deploy vineyardd --vineyardd-memory=4Gi

Prepare the S3 Service
----------------------

1. Deploy the Minio cluster:

   .. tip::

       If you already have the AWS S3 service, just skip this section and jump to
       the next section.

   .. code:: bash

       $ kubectl apply -f python/vineyard/contrib/kedro/benchmark/minio-s3/minio-dev.yaml

   .. tip::

       The default access key and secret key of the minio cluster are :code:`minioadmin`
       and :code:`minioadmin`.

2. Create the S3 bucket:

   - If you are working with AWS S3, you can create a bucket named
     :code:`aws-s3-benchmark-bucket` with the following command:

     .. code:: bash

         $ aws s3api create-bucket --bucket aws-s3-benchmark-bucket --region <Your AWS Region Name>

   - If you are working with Minio, you first need to expose the services
     and then create the bucket:

     - Forward minio-artifacts service:

       .. code:: bash

           $ kubectl port-forward service/minio -n minio-dev 9000:9000

     - Install the minio client:

       .. code:: bash

           $ wget https://dl.min.io/client/mc/release/linux-amd64/mc
           $ chmod +x mc
           $ sudo mv mc /usr/local/bin

     - Configure the minio client:

       .. code:: bash

           $ mc alias set minio http://localhost:9000
           Enter Access Key: <Your Access Key>
           Enter Secret Key: <Your Secret Key>

     - Finally, create the bucket :code:`minio-s3-benchmark-bucket`:

       .. code:: bash

           $ mc mb minio/minio-s3-benchmark-bucket
           Bucket created successfully `minio/minio-s3-benchmark-bucket`.

Prepare the Docker images
-------------------------

1. Vineyard has delivered `a benchmark project <https://github.com/v6d-io/v6d/tree/main/python/vineyard/contrib/kedro/benchmark>`_
   to test Kedro pipelines on Vineyard and S3:

   .. code:: bash

       $ cd python/vineyard/contrib/kedro/benchmark

2. Configure the credentials configurations of AWS S3:

   .. code:: bash

       $ cd aws-s3
       $ cat conf/local/credentials.yml
       benchmark_aws_s3:
           client_kwargs:
               aws_access_key_id: Your AWS/Minio Access Key ID
               aws_secret_access_key: Your AWS/Minio Secret Access Key
               region_name: Your AWS Region Name

3. To deploy pipelines to Kubernetes, you first need to build the Docker image for the
   benchmark project.

   To show how vineyard can accelerate the data sharing along with the dataset
   scales, Docker images for different data size will be generated:

   - For running Kedro on vineyard:

     .. code:: bash

         $ make -C vineyard/

     You will see Docker images for different data size are generated:

     .. code:: bash

         $ docker images | grep vineyard-benchmark
         vineyard-benchmark-with-500m-data   latest  982c6a376597   About a minute ago   1.66GB
         vineyard-benchmark-with-100m-data   latest  e58ca1cada98   About a minute ago   1.25GB
         vineyard-benchmark-with-10m-data    latest  f7c618b48913   About a minute ago   1.16GB
         vineyard-benchmark-with-1m-data     latest  8f9e74ff5116   About a minute ago   1.15GB

   - Similarly, for running Kedro on AWS S3 or Minio:

     .. code:: bash

         # for AWS S3
         $ make -C aws-s3/
         # for Minio
         $ make -C minio-s3/

4. To make those images available for your Kubernetes cluster, they need to be
   pushed to your registry (or load to kind cluster if you setup your Kubernetes
   cluster using kind):

   - Push to registry:

     .. code:: bash

         # for vineyard
         $ for sz in 1m 10m 100m 500m; do \
               docker tag vineyard-benchmark-with-${sz}-data <Your Registry>/vineyard-benchmark-with-${sz}-data; \
               docker push <Your Registry>/vineyard-benchmark-with-${sz}-data; \
           done

         # for AWS S3
         $ for sz in 1m 10m 100m 500m; do \
               docker tag aws-s3-benchmark-with-${sz}-data <Your Registry>/vineyard-benchmark-with-${sz}-data; \
               docker push <Your Registry>/vineyard-benchmark-with-${sz}-data; \
           done

         # for Minio
         $ for sz in 1m 10m 100m 500m; do \
               docker tag minio-s3-benchmark-with-${sz}-data <Your Registry>/vineyard-benchmark-with-${sz}-data; \
               docker push <Your Registry>/vineyard-benchmark-with-${sz}-data; \
           done

   - Load to kind cluster:

     .. code:: bash

         # for vineyard
         $ for sz in 1m 10m 100m 500m; do \
               kind load docker-image vineyard-benchmark-with-${sz}-data; \
           done

         # for AWS S3
         $ for sz in 1m 10m 100m 500m; do \
               kind load docker-image aws-s3-benchmark-with-${sz}-data; \
           done

         # for Minio
         $ for sz in 1m 10m 100m 500m; do \
               kind load docker-image minio-s3-benchmark-with-${sz}-data; \
           done

Deploy the Kedro Pipelines
--------------------------

1. Deploy the Kedro pipeline with vineyard for intermediate data sharing:

   .. code:: bash

       $ pushd vineyard
       $ kubectl create namespace vineyard

       $ for sz in 1m 10m 100m 500m; do \
             sed -i "s/vineyard-benchmark/vineyard-benchmark-with-${sz}-data/g" argo-vineyard-benchmark.yml && \
             argo submit -n vineyard --watch argo-vineyard-benchmark.yml; \
         done

       $ popd

2. Similarly, using AWS S3 or Minio for intermediate data sharing:

   - Using AWS S3:

     .. code:: bash

         $ pushd vineyard
         $ kubectl create namespace aws-s3

         $ for sz in 1m 10m 100m 500m; do \
               sed -i "s/aws-s3-benchmark/aws-s3-benchmark-with-${sz}-data/g" argo-aws-s3-benchmark.yml && \
               argo submit -n aws-s3 --watch argo-aws-s3-benchmark.yml; \
           done

         $ popd

   - Using Minio:

     .. code:: bash

         $ pushd aws-s3
         $ kubectl create namespace minio-s3

         $ for sz in 1m 10m 100m 500m; do \
               sed -i "s/minio-s3-benchmark/minio-s3-benchmark-with-${sz}-data/g" argo-minio-s3-benchmark.yml && \
               argo submit -n minio-s3 --watch argo-minio-s3-benchmark.yml; \
           done

         $ popd

Performance
-----------

After running the benchmark above on Kubernetes, we recorded the following end-to-end execution time under
different settings:

=========    =========    =========    =========
Data size    Vineyard     AWS S3       Minio S3
=========    =========    =========    =========
1M                 30s          50s          30s
10M                30s          63s          30s
100M               60s         144s          64s
500M               91s         457s         177s
=========    =========    =========    =========

We have the following observations from above comparison:

- Vineyard can significantly accelerate the data sharing between tasks in Kedro pipelines, without the
  need for any intrusive changes to the original Kedro pipelines;
- When data scales, the performance of Vineyard is more impressive, as the intermediate data sharing
  cost becomes more dominant in end-to-end execution;
- Even compared with local Minio, Vineyard still outperforms it by a large margin, thanks to the ability
  of Vineyard to avoid (de)serialization, file I/O and excessive memory copies.