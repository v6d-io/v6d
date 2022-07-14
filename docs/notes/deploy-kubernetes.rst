Deploying on Kubernetes
=======================

.. _deploying-on-kubernetes:

For better leveraging the scale-in/out capability of Kubernetes for worker pods of
a data analytical job, vineyard could be deployed on Kubernetes to as a DaemonSet
in Kubernetes cluster.

Cross-Pod IPC
^^^^^^^^^^^^^

Vineyard pods shares memory with worker pods using a UNIX domain socket with fine-grained
access control.

The UNIX domain socket can be either mounted on ``hostPath`` or via a ``PersistentVolumeClaim``.
When users bundle vineyard and the workload to the same pod, the UNIX domain socket
could also be shared using an ``emptyDir``.

**Note** that when deploying vineyard on Kubernetes, it usually can only be connected
from containers in the same pod, or pods on the same hosts.

Deploying with Helm
^^^^^^^^^^^^^^^^^^^

Vineyard also has tight integration with Kubernetes and Helm. Vineyard can be deployed
with ``helm``:

.. code:: shell

    helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
    helm install vineyard vineyard/vineyard

In the further vineyard will improve the integration with Kubernetes by abstract
vineyard objects as as Kubernetes resources (i.e., CRDs), and leverage a vineyard
operator to operate vineyard cluster.

Deploying using Python API
^^^^^^^^^^^^^^^^^^^^^^^^^^

For environments where the Helm is not available, vineyard provides a Python API for
deploying on Kubernetes,

.. code:: python

    from vineyard.deploy.kubernetes import start_vineyardd

    resources = start_vineyardd()

The deployed etcd pods, vineyard daemonset and services can be viewed as:

.. code:: shell

    $ kubectl -n vineyard get pods
    NAME             READY   STATUS    RESTARTS   AGE
    etcd0            1/1     Running   0          10s
    vineyard-pwkcn   1/1     Running   0          10s

    $ kubectl -n vineyard get daemonsets
    NAME       DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
    vineyard   1         1         1       1            1           <none>          14s

    $ kubectl -n vineyard get services
    NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
    etcd-for-vineyard   ClusterIP   ........        <none>        2379/TCP   18s
    vineyard-rpc        ClusterIP   ........        <none>        9600/TCP   18s

Later the cluster can be stoped by releasing the resources with

.. code:: python

    from vineyard.deploy.kubernetes import delete_kubernetes_objects

    delete_kubernetes_objects(resources)

For more details about the API usage, please refer to the :ref:`vineyard-python-deployment-api`.
