.. _accelerate-data-sharing-in-kedro:

Accelerate Data Sharing in Kedro
================================

This is a tutorial that shows how Vineyard accelerate the intermediate data
sharing between tasks in Kedro pipelines using our
`vineyard-kedro <https://pypi.org/project/vineyard-kedro/>`_ plugin, when data
scales and the pipeline are deployed on Kubernetes.

.. note::

    This tutorial is based on the `Developing and Learning MLOps at Home <https://github.com/AdamShafi92/mlops-at-home>`_ project,
    a tutorial about orchestrating a machine learning pipeline with Kedro.

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

       To handle the large data, we set the memory of vineyard cluster to 8G and 
        the shared memory to 8G.

   .. code:: bash

       $ python3 -m vineyard.ctl deploy vineyardd --vineyardd.memory=8Gi --vineyardd.size=8Gi

   .. note::

       The above command will try to create a vineyard cluster with 3 replicas
       by default. If you are working with Minikube, Kind, or other Kubernetes
       that has less nodes available, try reduce the replicas by

       .. code:: bash

           $ python3 -m vineyard.ctl deploy vineyardd --replicas=1 --vineyardd.memory=8Gi --vineyardd.size=8Gi

Prepare the S3 Service
----------------------

1. Deploy the Minio cluster:

   .. tip::

       If you already have the AWS S3 service, just skip this section and jump to
       the next section.

   .. code:: bash

       $ kubectl apply -f python/vineyard/contrib/kedro/benchmark/mlops/minio-dev.yaml

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

       $ cd python/vineyard/contrib/kedro/benchmark/mlops

2. Configure the credentials configurations of AWS S3:

   .. code:: bash

       $ cat conf/aws-s3/credentials.yml
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

         $ make docker-build

     You will see Docker images for different data size are generated:

     .. code:: bash

         $ docker images | grep mlops
         mlops-benchmark    latest    fceaeb5a6688   17 seconds ago   1.07GB

4. To make those images available for your Kubernetes cluster, they need to be
   pushed to your registry (or load to kind cluster if you setup your Kubernetes
   cluster using kind):

   - Push to registry:

     .. code:: bash

         $ docker tag mlops-benchmark:latest <Your Registry>/mlops-benchmark:latest
         $ docker push <Your Registry>/mlops-benchmark:latest

   - Load to kind cluster:

     .. code:: bash

         $ kind load docker-image mlops-benchmark:latest

Deploy the Kedro Pipelines
--------------------------

1. Deploy the Kedro pipeline with vineyard for intermediate data sharing:

   .. code:: bash

       $ kubectl create namespace vineyard
       $ for multiplier in 1 10 100 500; do \
            argo submit -n vineyard --watch argo-vineyard-benchmark.yml -p multiplier=${multiplier}; \
         done

2. Similarly, using AWS S3 or Minio for intermediate data sharing:

   - Using AWS S3:

     .. code:: bash

         $ kubectl create namespace aws-s3
         # create the aws secrets from your ENV
         $ kubectl create secret generic aws-secrets -n aws-s3 \
              --from-literal=access_key_id=$AWS_ACCESS_KEY_ID \
              --from-literal=secret_access_key=$AWS_SECRET_ACCESS_KEY
         $ for multiplier in 1 10 100 500 1000 2000; do \
              argo submit -n aws-s3 --watch argo-aws-s3-benchmark.yml -p multiplier=${multiplier}; \
           done

   - Using `Cloudpickle dataset <https://github.com/getindata/kedro-sagemaker/blob/dbd78fd6c1781cc9e8cf046e14b3ab96faf63719/kedro_sagemaker/datasets.py#L126>`_:

     .. code:: bash

         $ kubectl create namespace cloudpickle
         # create the aws secrets from your ENV
         $ kubectl create secret generic aws-secrets -n cloudpickle \
              --from-literal=access_key_id=$AWS_ACCESS_KEY_ID \
              --from-literal=secret_access_key=$AWS_SECRET_ACCESS_KEY
         $ for multiplier in 1 10 100 500 1000 2000; do \
              argo submit -n cloudpickle --watch argo-cloudpickle-benchmark.yml -p multiplier=${multiplier}; \
           done

   - Using Minio:

     .. code:: bash

         $ kubectl create namespace minio-s3
         $ for multiplier in 1 10 100 500 1000 2000; do \
              argo submit -n minio-s3 --watch argo-minio-s3-benchmark.yml -p multiplier=${multiplier}; \
           done

Performance
-----------

After running the benchmark above on Kubernetes, we recorded each node's execution time from the logs
of the argo workflow and calculated the sum of all nodes as the following end-to-end execution time 
for each data scale:

==========    =========    ========    ==============    =========
Data Scale    Vineyard     Minio S3    Cloudpickle S3     AWS S3
==========    =========    ========    ==============    =========
1                  4.2s        4.3s             22.5s        16.9s
10                 4.9s        5.5s             28.6s        23.3s
100               13.2s       20.3s             64.4s          74s
500               53.6s       84.5s            173.2s       267.9s
1000             109.8s      164.2s            322.7s       510.6s
2000             231.6s      335.9s            632.8s      1069.7s
==========    =========    ========    ==============    =========

We have the following observations from above comparison:

- Vineyard can significantly accelerate the data sharing between tasks in Kedro pipelines, without the
  need for any intrusive changes to the original Kedro pipelines;
- When data scales, the performance of Vineyard is more impressive, as the intermediate data sharing
  cost becomes more dominant in end-to-end execution;
- Even compared with local Minio, Vineyard still outperforms it by a large margin, thanks to the ability
  of Vineyard to avoid (de)serialization, file I/O and excessive memory copies.
- When using the Cloudpickle dataset(pickle + zstd), the performance is better than AWS S3, as the dataset
  will be compressed before uploading to S3.
