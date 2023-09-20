Efficient data sharing in Kubeflow with Vineyard CSI Driver
===========================================================

If you are using `Kubeflow Pipeline`_ or `Argo Workflow`_ to manage your machine learning workflow, 
you may find that the data saving/loading to the volumes is slow.
To speed up the data saving/loading within these volumes, we design the Vineyard CSI Driver to
map each vineyard object to a volume, and the data saving/loading is handled by vineyard.
Next, we will show you how to use the Vineyard CSI Driver to speed up a kubeflow pipeline.

Prerequisites
-------------

- A kubernetes cluster with version >= 1.25.10. If you don't have one by hand, you can refer to the 
guide `Initialize Kubernetes Cluster`_ to create one.
- Install the `Vineyardctl`_ by following the official guide.
- Install the `Argo Workflow CLI`_ by following the official guide.

Deploy the Vineyard Cluster
---------------------------

.. code:: bash

    $ vineyardctl deploy vineyard-cluster --create-namespace

This command will create a vineyard cluster in the namespace `vineyard-system`.
You can check as follows:

.. code:: bash

    $ kubectl get pod -n vineyard-system
    NAME                                             READY   STATUS    RESTARTS   AGE
    vineyard-controller-manager-648fc9b7bf-zwnhd     2/2     Running   0          4d3h
    vineyardd-sample-79c8ffb879-6k8mk                1/1     Running   0          4d3h
    vineyardd-sample-79c8ffb879-f9kkr                1/1     Running   0          4d3h
    vineyardd-sample-79c8ffb879-lzgwz                1/1     Running   0          4d3h
    vineyardd-sample-etcd-0                          1/1     Running   0          4d3h

Deploy the Vineyard CSI Driver
------------------------------

Before deploying the Vineyard CSI Driver, you are supposed to check the vineyard 
deployment is ready as follows:

.. code:: bash

    $ kubectl get deployment -n vineyard-system        
    NAME                             READY   UP-TO-DATE   AVAILABLE   AGE
    vineyard-controller-manager      1/1     1            1           4d3h
    vineyardd-sample                 3/3     3            3           4d3h

Then deploy the vineyard csi driver which specifies the vineyard cluster to use:

.. code:: bash

    $ vineyardctl deploy csidriver --clusters vineyard-system/vineyardd-sample

Then check the status of the Vineyard CSI Driver:

.. code:: bash

    $ kubectl get pod -n vineyard-system
    NAME                                             READY   STATUS    RESTARTS   AGE
    vineyard-controller-manager-648fc9b7bf-zwnhd     2/2     Running   0          4d3h
    vineyard-csi-sample-csi-driver-fb7cb5b5d-nlrxs   4/4     Running   0          4m23s
    vineyard-csi-sample-csi-nodes-69j77              3/3     Running   0          4m23s
    vineyard-csi-sample-csi-nodes-k85hb              3/3     Running   0          4m23s
    vineyard-csi-sample-csi-nodes-zhfz4              3/3     Running   0          4m23s
    vineyardd-sample-79c8ffb879-6k8mk                1/1     Running   0          4d3h
    vineyardd-sample-79c8ffb879-f9kkr                1/1     Running   0          4d3h
    vineyardd-sample-79c8ffb879-lzgwz                1/1     Running   0          4d3h
    vineyardd-sample-etcd-0                          1/1     Running   0          4d3h

Deploy Argo Workflows
---------------------

Install the argo server on Kubernetes:

.. code:: bash

    $ kubectl create namespace argo
    $ kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.4.8/install.yaml

Then check the status of the argo server:

.. code:: bash

    $ kubectl get pod -n argo           
    NAME                                  READY   STATUS    RESTARTS   AGE
    argo-server-7698c96655-ft6sj          1/1     Running   0          4d1h
    workflow-controller-b888f4458-sfrjd   1/1     Running   0          4d1h

Running a Kubeflow Pipeline example
-----------------------------------

The example is under the directory ``k8s/examples/vineyard-csidriver``, and ``pipeline.py`` under this
directory is the original pipeline definition. To use the Vineyard CSI Driver, we need to do two 
modifications:

1. Change APIs like **pd.read_pickle/write_pickle** to **vineyard.csi.write/read** in the source code.

2. Add the ``vineyard object`` VolumeOp to the pipeline's dependencies. The path in the API changed 
in the first step will be mapped to a volume. Notice, the volume used in any task needs to be 
explicitly mounted to the corresponding path in the source code, and the storageclass_name 
format of each VolumeOp is ``{vineyard-deployment-namespace}.{vineyard-deployment-name}.csi``.

You may get some insights from the modified pipeline ``pipeline-with-vineyard.py``. Then, we need to
compile the pipeline to an argo-workflow yaml. To be compatible with benchmark test, we update the
generated ``pipeline.yaml`` and ``pipeline-with-vineyard.yaml``.

Now, we can build the docker images for the pipeline:

.. code:: bash

    $ cd k8s/examples/vineyard-csidriver
    $ make docker-build

Check the images built successfully:

.. code:: bash

    $ docker images
    train-data               latest    5628953ffe08   14 seconds ago   1.47GB
    test-data                latest    94c8c75b960a   14 seconds ago   1.47GB
    prepare-data             latest    5aab1b120261   15 seconds ago   1.47GB
    preprocess-data          latest    5246d09e6f5e   15 seconds ago   1.47GB

Then push the image to a docker registry that your kubernetes cluster can access, as
we use the kind cluster in this example, we can load the image to the clusters:

.. code:: bash

    $ make load-images

To simulate the data loading/saving of the actual pipeline, we use the nfs volume
to store the data. The nfs volume is mounted to the ``/mnt/data`` directory of the 
kind cluster. Then apply the data volume as follows:

.. tip::

    If you already have nfs volume that can be accessed by the kubernetes cluster,
    you can update the prepare-data.yaml to use your nfs volume.

.. code:: bash

    $ kubectl apply -f prepare-data.yaml

Deploy the rbac for the pipeline:

.. code:: bash

    $ kubectl apply -f rbac.yaml

Submit the kubeflow example without vineyard to the argo server:

.. code:: bash

    $ for data_multiplier in 3000 4000 5000; do \
        argo submit --watch pipeline.yaml -p data_multiplier=${data_multiplier}; \
    done

Clear the previous resources:

.. code:: bash

    $ argo delete --all

Submit the kubeflow example with vineyard to the argo server:

.. code:: bash

    $ for data_multiplier in 3000 4000 5000; do \
        argo submit --watch pipeline-with-vineyard.yaml -p data_multiplier=${data_multiplier}; \
    done

Result Analysis
---------------

+------------+------------------+---------------+
| data scale | without vineyard | with vineyard |
+============+==================+===============+
| 8500 Mi    | 21s              | 5.4s          |
+------------+------------------+---------------+
| 12000 Mi   | 26s              | 7s            |
+------------+------------------+---------------+
| 15000 Mi   | 32.2s            | 9.4s          |
+------------+------------------+---------------+

The data scale are 8500 Mi, 12000 Mi and 15000 Mi, which correspond to 
the 3000, 4000 and 5000 in the previous data_multiplier respectively, 
and the time of argo workflow execution of the pipeline is as follows:

Argo workflow duration
======================

+------------+------------------+---------------+
| data scale | without vineyard | with vineyard |
+============+==================+===============+
| 8500 Mi    | 186s             | 169s          |
+------------+------------------+---------------+
| 12000 Mi   | 250s             | 203s          |
+------------+------------------+---------------+
| 15000 Mi   | 332s             | 286s          |
+------------+------------------+---------------+


Actually, the cost time of argo workflow is affected by lots of factors, 
such as the network, the cpu and memory of the cluster, the data volume, etc.
So the time of argo workflow execution of the pipeline is not stable. 
But we can still find that the time of argo workflow execution of the pipeline
with vineyard is shorter than that without vineyard.

Also, we record the whole execution time via logs. The result is as follows:

Actual execution time
=====================

+------------+------------------+---------------+
| data scale | without vineyard | with vineyard |
+============+==================+===============+
| 8500 Mi    | 139.3s           | 92.3s         |
+------------+------------------+---------------+
| 12000 Mi   | 204.3s           | 131.1s        |
+------------+------------------+---------------+
| 15000 Mi   | 289.3s           | 209.7s        |
+------------+------------------+---------------+


According to the above results, we can find that the time of actual 
execution of the pipeline with vineyard is shorter than that without vineyard.
To be specific, we record the write/read time of the following steps:

Writing time
============

+------------+------------------+---------------+
| data scale | without vineyard | with vineyard |
+============+==================+===============+
| 8500 Mi    | 21s              | 5.4s          |
+------------+------------------+---------------+
| 12000 Mi   | 26s              | 7s            |
+------------+------------------+---------------+
| 15000 Mi   | 32.2s            | 9.4s          |
+------------+------------------+---------------+


From the above results, we can find that the writing time the pipeline 
with vineyard is nearly 4 times shorter than that without vineyard. 
The reason is that the data is stored in the vineyard cluster, 
so it's actually a memory copy operation, which is faster than the 
write operation of the nfs volume.


Reading time
============

We delete the time of init data loading, and the results are as follows:

+------------+------------------+---------------+
| data scale | without vineyard | with vineyard |
+============+==================+===============+
| 8500 Mi    | 36.7s            | 0.02s         |
+------------+------------------+---------------+
| 12000 Mi   | 45.7s            | 0.02s         |
+------------+------------------+---------------+
| 15000 Mi   | 128.6s           | 0.04s         |
+------------+------------------+---------------+

Based on the above results, we can find that the read time of vineyard is
nearly a constant, which is not affected by the data scale.
The reason is that the data is stored in the shared memory of vineyard cluster, 
so it's actually a pointer copy operation.

As a result, we can find that with vineyard, the argo workflow 
duration of the pipeline is reduced by 10%~20% and the actual 
execution time of the pipeline is reduced by about 30%.


Clean up
--------

Delete the rbac for the kubeflow example:

.. code:: bash

    $ kubectl delete -f rbac.yaml

Delete all argo workflow

.. code:: bash

    $ argo delete --all

Delete the argo server:

.. code:: bash

    $ kubectl delete ns argo

Delete the csi driver:

.. code:: bash

    $ vineyardctl delete csidriver

Delete the vineyard cluster:

.. code:: bash

    $ vineyardctl delete vineyard-cluster

Delete the data volume:

.. code:: bash

    $ kubectl delete -f prepare-data.yaml

.. _Kubeflow Pipeline: https://github.com/kubeflow/kubeflow
.. _Argo Workflow: https://github.com/argoproj/argo-workflows
.. _Initialize Kubernetes Cluster: https://v6d.io/tutorials/kubernetes/using-vineyard-operator.html#step-0-optional-initialize-kubernetes-cluster
.. _Vineyardctl: https://v6d.io/notes/developers/build-from-source.html#install-vineyardctl
.. _Argo Workflow CLI: https://github.com/argoproj/argo-workflows/releases/