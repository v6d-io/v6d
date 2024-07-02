Use vineyard operator
=====================

Vineyard operator has been designed to manage vineyard components within Kubernetes.
This tutorial provides a step-by-step guide on how to effectively utilize the vineyard
operator. For more details, please refer to :ref:`vineyard-operator`.

Step 0: (optional) Initialize Kubernetes Cluster
------------------------------------------------

If you don't have a Kubernetes cluster readily available, we highly recommend using `kind`_ to
create one. Before setting up the Kubernetes cluster, please ensure you have the following
tools installed:

- kubectl: version >= 1.19.2
- kind: version >= 0.14.0
- docker: version >= 0.19.0

Utilize kind (v0.14.0) to create a Kubernetes cluster consisting of 4 nodes (1 master node and 3
worker nodes):

.. code:: bash

    $ cat > kind-config.yaml <<EOF
    # four node (three workers) cluster config
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    nodes:
    - role: control-plane
    - role: worker
    - role: worker
    - role: worker
    EOF
    $ kind create cluster --config kind-config.yaml

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

       Creating cluster "kind" ...
        ✓ Ensuring node image (kindest/node:v1.24.0) 🖼
        ✓ Preparing nodes 📦
        ✓ Writing configuration 📜
        ✓ Starting control-plane 🕹️
        ✓ Installing CNI 🔌
        ✓ Installing StorageClass 💾
        Set kubectl context to "kind-kind"
        You can now use your cluster with:

        kubectl cluster-info --context kind-kind

        Have a question, bug, or feature request? Let us know! https://kind.sigs.k8s.io/#community 🙂

.. note::

    The kind cluster's config file is stored in ~/.kube/config, so you can use
    the kubectl directly as it's the default config path.

Check all kubernetes pods.

.. code:: bash

    $ kubectl get pod -A

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAMESPACE            NAME                                         READY   STATUS    RESTARTS   AGE
        kube-system          coredns-6d4b75cb6d-k2sk2                     1/1     Running   0          38s
        kube-system          coredns-6d4b75cb6d-xm4dt                     1/1     Running   0          38s
        kube-system          etcd-kind-control-plane                      1/1     Running   0          52s
        kube-system          kindnet-fp24b                                1/1     Running   0          19s
        kube-system          kindnet-h6swp                                1/1     Running   0          39s
        kube-system          kindnet-mtkd4                                1/1     Running   0          19s
        kube-system          kindnet-zxxpd                                1/1     Running   0          19s
        kube-system          kube-apiserver-kind-control-plane            1/1     Running   0          52s
        kube-system          kube-controller-manager-kind-control-plane   1/1     Running   0          53s
        kube-system          kube-proxy-6zgq2                             1/1     Running   0          19s
        kube-system          kube-proxy-8vghn                             1/1     Running   0          39s
        kube-system          kube-proxy-c7vz5                             1/1     Running   0          19s
        kube-system          kube-proxy-kd4zz                             1/1     Running   0          19s
        kube-system          kube-scheduler-kind-control-plane            1/1     Running   0          52s
        local-path-storage   local-path-provisioner-9cd9bd544-2vrtq       1/1     Running   0          38s

Check all kubernetes nodes.

.. code:: bash

    $ kubectl get nodes -A

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                 STATUS   ROLES           AGE     VERSION
        kind-control-plane   Ready    control-plane   2m30s   v1.24.0
        kind-worker          Ready    <none>          114s    v1.24.0
        kind-worker2         Ready    <none>          114s    v1.24.0
        kind-worker3         Ready    <none>          114s    v1.24.0

Step 1: Deploy the Vineyard Operator
-------------------------------------

Create a dedicated namespace for the Vineyard Operator.

.. code:: bash

    $ kubectl create namespace vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        namespace/vineyard-system created

The Vineyard CRDs、Controllers、Webhooks and Scheduler are packaged by `helm`_, you could
deploy all resources as follows.

.. note::

    The vineyard operator needs permission to create several CRDs and kubernetes
    resources, before deploying the vineyard operator, please ensure you can create
    the `clusterrole`_. 

.. code:: bash

    $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        "vineyard" has been added to your repositories

Update the vineyard operator chart to the newest one.

.. code:: bash

    $ helm repo update

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        Hang tight while we grab the latest from your chart repositories...
         ...Successfully got an update from the "vineyard" chart repository
        Update Complete. ⎈Happy Helming!⎈

Deploy the vineyard operator in the namespace ``vineyard-system``.

.. code:: bash

    $ helm install vineyard-operator vineyard/vineyard-operator -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME: vineyard-operator
        LAST DEPLOYED: Wed Jan  4 16:41:45 2023
        NAMESPACE: vineyard-system
        STATUS: deployed
        REVISION: 1
        TEST SUITE: None
        NOTES:
        Thanks for installing VINEYARD-OPERATOR, release at namespace: vineyard-system, name: vineyard-operator.

        To learn more about the release, try:

        $ helm status vineyard-operator -n vineyard-system   # get status of running vineyard operator
        $ helm get all vineyard-operator -n vineyard-system  # get all deployment yaml of vineyard operator

        To uninstall the release, try:

        $ helm uninstall vineyard-operator -n vineyard-system

You could get all details about vineyard operator in the doc :ref:`vineyard-operator`, just have fun with vineyard operator!

Check the status of all vineyard resources created by helm:

.. code:: bash

    $ kubectl get all -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                                                            READY   STATUS    RESTARTS   AGE
        pod/vineyard-operator-controller-manager-5bcbb75fb6-cfdpk       2/2     Running   0          2m30s

        NAME                                                           TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
        service/vineyard-operator-controller-manager-metrics-service   ClusterIP   10.96.153.134   <none>        8443/TCP   2m30s
        service/vineyard-operator-webhook-service                      ClusterIP   10.96.9.101     <none>        443/TCP    2m30s

        NAME                                                        READY   UP-TO-DATE   AVAILABLE   AGE
        deployment.apps/vineyard-operator-controller-manager        1/1     1            1           2m30s

        NAME                                                                  DESIRED   CURRENT   READY   AGE
        replicaset.apps/vineyard-operator-controller-manager-5bcbb75fb6       1         1         1       2m30s

Step 2: Deploy a Vineyard Cluster
----------------------------------

After successfully installing the Vineyard operator as described in the previous step,
you can now proceed to deploy a Vineyard cluster. To create a cluster with two Vineyard
instances, simply create a `Vineyardd` Custom Resource (CR) as shown below.

.. code:: bash

    $ cat <<EOF | kubectl apply -f -
    apiVersion: k8s.v6d.io/v1alpha1
    kind: Vineyardd
    metadata:
      name: vineyardd-sample
      namespace: vineyard-system
    spec:
      # vineyard instances
      replicas: 2
    EOF

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        vineyardd.k8s.v6d.io/vineyardd-sample created

Check the status of all relevant resources managed by the ``vineyardd-sample`` cr.

.. code:: bash

    $ kubectl get all -l app.kubernetes.io/instance=vineyard-system-vineyardd-sample -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                                   READY   STATUS    RESTARTS   AGE
        pod/vineyardd-sample-879798cb6-qpvtw   1/1     Running   0          2m59s
        pod/vineyardd-sample-879798cb6-x4m2x   1/1     Running   0          2m59s

        NAME                               READY   UP-TO-DATE   AVAILABLE   AGE
        deployment.apps/vineyardd-sample   2/2     2            2           2m59s

        NAME                                         DESIRED   CURRENT   READY   AGE
        replicaset.apps/vineyardd-sample-879798cb6   2         2         2       2m59s

Step 3: Connect to Vineyard
----------------------------

Vineyard currently supports clients in various languages, including mature support
for C++ and Python, as well as experimental support for Java, Golang, and Rust. In
this tutorial, we will demonstrate how to connect to a Vineyard cluster using the
Python client. Vineyard provides two connection methods: `IPC and RPC`_. In the
following sections, we will explore both methods.

First, let's deploy the Python client on two Vineyard nodes as follows.

.. code:: bash

    $ cat <<EOF | kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vineyard-python-client
      namespace:  vineyard-system
    spec:
      selector:
        matchLabels:
          app: vineyard-python-client
      replicas: 2
      template:
        metadata:
          labels:
            app: vineyard-python-client
            # related to which vineyard cluster
            scheduling.k8s.v6d.io/vineyardd-namespace: vineyard-system
            scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
            scheduling.k8s.v6d.io/job: v6d-workflow-demo-job1
        spec:
          # use the vineyard scheduler to deploy the pod on the vineyard cluster.
          schedulerName: vineyard-scheduler
          containers:
          - name: vineyard-python
            imagePullPolicy: IfNotPresent
            image: python:3.10
            command:
            - /bin/bash
            - -c
            - pip3 install vineyard && sleep infinity
            volumeMounts:
            - mountPath: /var/run
              name: vineyard-sock
          volumes:
          - name: vineyard-sock
            hostPath:
              path: /var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample
    EOF

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        pod/vineyard-python-client created

Wait for the vineyard python client pod ready.

.. code:: bash

    $ kubectl get pod -l app=vineyard-python-client -n vineyard-system


.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        NAME                                      READY   STATUS    RESTARTS   AGE
        vineyard-python-client-6fd84bc897-27glp   1/1     Running   0          93s
        vineyard-python-client-6fd84bc897-tlb22   1/1     Running   0          93s

Use the kubectl exec command to enter the first vineyard python client pod.

.. code:: bash

    $ kubectl exec -it $(kubectl get pod -l app=vineyard-python-client -n vineyard-system -oname | head -n 1 | awk -F '/' '{print $2}') -n vineyard-system /bin/bash

After entering the shell, you can connect to the vineyard cluster,

.. code-block:: python

    In [1]: import numpy as np
    In [2]: import vineyard

    In [3]: client = vineyard.connect('/var/run/vineyard.sock')

    In [4]: objid = client.put(np.zeros(8))

    In [5]: # persist the object to make it visible to form the global object
    In [6]: client.persist(objid)

    In [7]: objid
    Out[7]: o001027d7c86a49f0

    In [8]: # get meta info
    In [9]: meta = client.get_meta(objid)
    In [10]: meta
    Out[10]:
    {
        "buffer_": {
            "id": "o801027d7c85c472e",
            "instance_id": 1,
            "length": 0,
            "nbytes": 0,
            "transient": true,
            "typename": "vineyard::Blob"
        },
        "global": false,
        "id": "o001027d7c86a49f0",
        "instance_id": 1,
        "nbytes": 64,
        "order_": "\"C\"",
        "partition_index_": "[]",
        "shape_": "[8]",
        "signature": 4547407361228035,
        "transient": false,
        "typename": "vineyard::Tensor<double>",
        "value_type_": "float64",
        "value_type_meta_": "<f8"
    }

Open another terminal and enter the second vineyard python client pod.

.. code:: bash

    $ kubectl exec -it $(kubectl get pod -l app=vineyard-python-client -n vineyard-system -oname | tail -n 1 | awk -F '/' '{print $2}') -n vineyard-system /bin/bash

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.

Also, you can connect to the vineyard cluster by RPC and get the metadata of
above object as follows.

.. code-block:: python

    In [1]: import vineyard

    In [2]: rpc_client = vineyard.connect('vineyardd-sample-rpc.vineyard-system',9600)

    In [3]: # use the object id created by another vineyard instance here
    In [4]: meta = rpc_client.get_meta(vineyard._C.ObjectID('o001027d7c86a49f0'))
    In [5]: meta
    Out[5]:
    {
        "buffer_": {
            "id": "o801027d7c85c472e",
            "instance_id": 1,
            "length": 0,
            "nbytes": 0,
            "transient": true,
            "typename": "vineyard::Blob"
        },
        "global": false,
        "id": "o001027d7c86a49f0",
        "instance_id": 1,
        "nbytes": 64,
        "order_": "\"C\"",
        "partition_index_": "[]",
        "shape_": "[8]",
        "signature": 4547407361228035,
        "transient": false,
        "typename": "vineyard::Tensor<double>",
        "value_type_": "float64",
        "value_type_meta_": "<f8"
    }

For more examples, please refer the `vineyard data accessing`_.

Step 4: Cleanup
---------------

- Destroy the vineyard operator via helm:

.. code:: bash

    $ helm uninstall vineyard-operator -n vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        release "vineyard-operator" uninstalled

- Delete the namespace:

.. code:: bash

    $ kubectl delete namespace vineyard-system

.. admonition:: Expected output
   :class: admonition-details

    .. code:: bash

        namespace "vineyard-system" deleted


.. _kind: https://kind.sigs.k8s.io
.. _helm: https://helm.sh/docs/intro/install/
.. _IPC and RPC: https://v6d.io/notes/key-concepts/data-accessing.html#ipcclient-vs-rpcclient
.. _vineyard data accessing: https://v6d.io/notes/key-concepts/data-accessing.html#data-accessing
.. _clusterrole: https://github.com/v6d-io/v6d/blob/main/k8s/config/rbac/role.yaml