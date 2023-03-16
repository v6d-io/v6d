.. _deploy-on-kubernetes:

Deploy on Kubernetes
====================

Vineyard is managed by the :ref:`vineyard-operator` on Kubernetes.

Install vineyard-operator
-------------------------

There are two ways to install vineyard operator: installing using Helm(recommended), or
installing from source code.

.. note::

    Before you install vineyard operator, you should have a Kubernetes cluster and kubectl
    installed. Here we use `kind`_ to create a cluster.

Before installing vineyard, you need ensure cert-manager is installed first to meet the
requirements of the webhook components in vineyard operator:

Install cert-manager
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    $ kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml

.. note::

    Please wait the cert-manager for a while until it is ready before installing the
    vineyard operator.

Option #1: install from helm chart (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
    $ helm install vineyard-operator vineyard/vineyard-operator

Wait the vineyard operator until ready:

Option #2: Install form source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the vineyard repo:

   .. code:: bash

      $ git clone https://github.com/v6d-io/v6d.git

2. Build the vineyard operator's Docker image:

   .. code:: bash

      $ cd k8s
      $ make -C k8s docker-build

   With `kind`_, you need to first import the image into the kind cluster (otherwise you
   need to push the image to your registry first):

   .. code:: bash

      $ kind load docker-image vineyardcloudnative/vineyard-operator:latest

3. Next, deploy the vineyard operator:

   .. code:: bash

      $ make -C k8s deploy

Wait vineyard-operator ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the operator installed, it's deployment can be checked using :code:`kubectl`:

.. code:: bash

    $ kubectl get all -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                                               READY   STATUS    RESTARTS   AGE
        pod/vineyard-controller-manager-5c6f4bc454-8xm8q   2/2     Running   0          62m

        NAME                                                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
        service/vineyard-controller-manager-metrics-service   ClusterIP   10.96.240.173   <none>        8443/TCP   62m
        service/vineyard-webhook-service                      ClusterIP   10.96.41.132    <none>        443/TCP    62m

        NAME                                          READY   UP-TO-DATE   AVAILABLE   AGE
        deployment.apps/vineyard-controller-manager   1/1     1            1           62m

        NAME                                                     DESIRED   CURRENT   READY   AGE
        replicaset.apps/vineyard-controller-manager-5c6f4bc454   1         1         1       62m

Create vineyard cluster
-----------------------

Once the vineyard operator become ready, you can create a vineyard cluster by creating a
:code:`Vineyardd` `CRD`_. The following is an example of creating a vineyard cluster with 3 daemon
replicas:

.. code:: yaml

    $ cat <<EOF | kubectl apply -f -
    apiVersion: k8s.v6d.io/v1alpha1
    kind: Vineyardd
    metadata:
      name: vineyardd-sample
      # don't use default namespace
      namespace: vineyard-system
    spec:
      replicas: 2
      etcd:
        replicas: 3
      service:
        type: ClusterIP
        port: 9600
      vineyardConfig:
        image: ghcr.io/v6d-io/v6d/vineyardd:alpine-latest
        imagePullPolicy: IfNotPresent
    EOF

The vineyard-operator will create required dependencies (e.g., etcd) a :code:`Deployment`` for
a 3-replicas vineyard servers. Once ready, you can inspect the components created and managed by
vineyard-operator using :code:`kubectl`:

.. code:: bash

    $ kubectl get all -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                                               READY   STATUS    RESTARTS   AGE
        pod/etcd0                                          1/1     Running   0          48s
        pod/etcd1                                          1/1     Running   0          48s
        pod/etcd2                                          1/1     Running   0          48s
        pod/vineyard-controller-manager-5c6f4bc454-8xm8q   2/2     Running   0          72s
        pod/vineyardd-sample-5cc797668f-9ggr9              1/1     Running   0          48s
        pod/vineyardd-sample-5cc797668f-nhw7p              1/1     Running   0          48s
        pod/vineyardd-sample-5cc797668f-r56h7              1/1     Running   0          48s

        NAME                                                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)             AGE
        service/etcd-for-vineyard                             ClusterIP   10.96.174.41    <none>        2379/TCP            48s
        service/etcd0                                         ClusterIP   10.96.128.87    <none>        2379/TCP,2380/TCP   48s
        service/etcd1                                         ClusterIP   10.96.72.116    <none>        2379/TCP,2380/TCP   48s
        service/etcd2                                         ClusterIP   10.96.99.182    <none>        2379/TCP,2380/TCP   48s
        service/vineyard-controller-manager-metrics-service   ClusterIP   10.96.240.173   <none>        8443/TCP            72s
        service/vineyard-webhook-service                      ClusterIP   10.96.41.132    <none>        443/TCP             72s
        service/vineyardd-sample-rpc                          ClusterIP   10.96.102.183   <none>        9600/TCP            48s

        NAME                                          READY   UP-TO-DATE   AVAILABLE   AGE
        deployment.apps/vineyard-controller-manager   1/1     1            1           72s
        deployment.apps/vineyardd-sample              3/3     3            3           48s

        NAME                                                     DESIRED   CURRENT   READY   AGE
        replicaset.apps/vineyard-controller-manager-5c6f4bc454   1         1         1       72s
        replicaset.apps/vineyardd-sample-5cc797668f              3         3         3       48s

References
----------

Besides deploying and managing vineyard cluster, the operator is responsible for scheduling workloads
on vineyard to optimize the data sharing between tasks in workflows and triggering required data movement/
transformation tasks as well, we list the detailed references and examples in :code:`vineyard-operator`.

To ease the interaction with vineyard on Kubernetes, we provide a command-line tool :code:`vineyardctl`
which automate many boilerplate configuration that required during deploying workflows with vineyard
on Kubernetes.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./vineyard-operator
      :type: ref
      :text: Vineyard operator
      :classes: btn-block stretched-link text-left
   ^^^^^^^^^^^^
   Vineyard operator manages vineyard cluster and orchestrates shared objects on Kubernetes.

   ---

   .. link-button:: ./vineyardctl
      :type: ref
      :text: vineyardctl
      :classes: btn-block stretched-link text-left
   ^^^^^^^^^^^^
   :code:`vineyardctl` is the command-line tool for working with the Vineyard Operator.

.. _kind: https://kind.sigs.k8s.io
.. _CRD: https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions
