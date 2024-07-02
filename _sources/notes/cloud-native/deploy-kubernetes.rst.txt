.. _deploy-on-kubernetes:

Deploy on Kubernetes
====================

Vineyard is managed by the :ref:`vineyard-operator` on Kubernetes.

Quick start
-----------

If you want to install vineyard cluster quickly, you can 
use the following command.

Install `vineyardctl`_ as follows.

.. code:: bash

    pip3 install vineyard

Use the vineyardctl to install vineyard cluster.

.. code:: bash

    python3 -m vineyard.ctl deploy vineyard-cluster --create-namespace

Also, you could follow the next guide to install vineyard cluster steps
by steps.

Install vineyard-operator
-------------------------

There are two recommended methods for installing the vineyard operator: using Helm (preferred) or
installing directly from the source code.

.. note::

    Prior to installing the vineyard operator, ensure that you have a Kubernetes cluster and kubectl
    installed. In this guide, we will use `kind`_ to create a cluster.


Option #1: Install from helm chart (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
    $ helm repo update
    $ helm install vineyard-operator vineyard/vineyard-operator \
          --namespace vineyard-system \
          --create-namespace

Wait for the vineyard operator until ready.

Option #2: Install form source code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the vineyard repo:

   .. code:: bash

      $ git clone https://github.com/v6d-io/v6d.git

2. Build the vineyard operator's Docker image:

   .. code:: bash

      $ cd k8s
      $ make -C k8s docker-build

   .. caution::

      With `kind`_, you need to first import the image into the kind cluster:

      .. code:: bash

          $ kind load docker-image vineyardcloudnative/vineyard-operator:latest

3. Next, deploy the vineyard operator:

   .. code:: bash

      $ make -C k8s deploy

Wait vineyard-operator ready
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the operator is installed, its deployment can be checked using :code:`kubectl`:

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

Once the vineyard operator becomes ready, you can create a vineyard cluster by creating a
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
      replicas: 3
      service:
        type: ClusterIP
        port: 9600
      vineyard:
        image: vineyardcloudnative/vineyardd:latest
        imagePullPolicy: IfNotPresent
    EOF

The vineyard-operator efficiently creates the necessary dependencies, such as etcd, and establishes a
:code:`Deployment` for a 3-replica vineyard server configuration. Once the setup is complete, you can
conveniently inspect the components created and managed by the vineyard operator using the :code:`kubectl`
command.

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

In addition to deploying and managing the vineyard cluster, the operator plays a crucial role in scheduling
workloads on vineyard. This optimizes data sharing between tasks in workflows and triggers necessary data
movement or transformation tasks. Detailed references and examples can be found in :code:`vineyard-operator`.

To simplify interactions with vineyard on Kubernetes, we offer a command-line tool, :code:`vineyardctl`, which
automates much of the boilerplate configuration required when deploying workflows with vineyard on Kubernetes.

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

.. _vineyardctl: https://github.com/v6d-io/v6d/blob/main/k8s/cmd/README.md
.. _kind: https://kind.sigs.k8s.io
.. _CRD: https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions
