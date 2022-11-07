Deploying on Kubernetes
=======================

.. _deploying-on-kubernetes:

Deploying with Vineyard Operator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please refer to `Vineyard Operator <https://github.com/v6d-io/v6d/blob/main/notes/vineyard-operator.rst>`_.

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
