How to use vineyard operator?
=============================

Vineyard (v6d) is an in-memory immutable data manager that provides out-of-the-box 
high-level abstraction and zero-copy in-memory sharing for distributed data in big 
data tasks, such as graph analytics (e.g., GraphScope), numerical computing 
(e.g., Mars), and machine learning. To manager vineyard components in Kubernetes 
cluster, we proposal the vineyard operator. For more details, you could refer the doc 
to get all features about vineyard operator. The blog will show you how to use the 
vineyard operator v0.11.4 hand by hand. 

1. [optional] Create a kubernetes cluster
-----------------------------------------

If you don't have a kubernetes cluster by hand, we highly recommend to use `kind`_ to 
create a kubernetes cluster. Before creating the kubernetes cluster, please make sure 
you have the following tools already:

- kubectl: version >= 1.19.2
- kind: version >= 0.14.0
- docker: version >= 0.19.0

Use kind (v0.14.0) to create a kubernetes cluster with 4 nodes(1 master nodes and 3 
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

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

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

.. raw:: html

   </details>

.. note::

    The kind cluster's config file is stored in ~/.kube/config, so you can use 
    the kubectl directly as it's the default config path.

Check all kubernetes pods.

.. code:: bash

    $ kubectl get pod -A

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

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

.. raw:: html

   </details>

Check all kubernetes nodes.

.. code:: bash

    $ kubectl get nodes -A

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME                 STATUS   ROLES           AGE     VERSION
    kind-control-plane   Ready    control-plane   2m30s   v1.24.0
    kind-worker          Ready    <none>          114s    v1.24.0
    kind-worker2         Ready    <none>          114s    v1.24.0
    kind-worker3         Ready    <none>          114s    v1.24.0

.. raw:: html

   </details>

2. Deploy vineyard operator
---------------------------

Create a namespace for vineyard operator.

.. code:: bash

    $ kubectl create namespace vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    namespace/vineyard-system created

.. raw:: html

   </details>

The operator needs a certificate created by cert-manager for webhook(https), 
so we have to install the cert-manager(v1.9.1) at first.

.. code:: bash

    $ kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    namespace/cert-manager created
    customresourcedefinition.apiextensions.k8s.io/certificaterequests.cert-manager.io created
    customresourcedefinition.apiextensions.k8s.io/certificates.cert-manager.io created
    customresourcedefinition.apiextensions.k8s.io/challenges.acme.cert-manager.io created
    customresourcedefinition.apiextensions.k8s.io/clusterissuers.cert-manager.io created
    customresourcedefinition.apiextensions.k8s.io/issuers.cert-manager.io created
    customresourcedefinition.apiextensions.k8s.io/orders.acme.cert-manager.io created
    serviceaccount/cert-manager-cainjector created
    serviceaccount/cert-manager created
    serviceaccount/cert-manager-webhook created
    configmap/cert-manager-webhook created
    clusterrole.rbac.authorization.k8s.io/cert-manager-cainjector created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-issuers created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-clusterissuers created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-certificates created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-orders created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-challenges created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-ingress-shim created
    clusterrole.rbac.authorization.k8s.io/cert-manager-view created
    clusterrole.rbac.authorization.k8s.io/cert-manager-edit created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-approve:cert-manager-io created
    clusterrole.rbac.authorization.k8s.io/cert-manager-controller-certificatesigningrequests created
    clusterrole.rbac.authorization.k8s.io/cert-manager-webhook:subjectaccessreviews created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-cainjector created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-issuers created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-clusterissuers created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-certificates created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-orders created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-challenges created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-ingress-shim created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-approve:cert-manager-io created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-controller-certificatesigningrequests created
    clusterrolebinding.rbac.authorization.k8s.io/cert-manager-webhook:subjectaccessreviews created
    role.rbac.authorization.k8s.io/cert-manager-cainjector:leaderelection created
    role.rbac.authorization.k8s.io/cert-manager:leaderelection created
    role.rbac.authorization.k8s.io/cert-manager-webhook:dynamic-serving created
    rolebinding.rbac.authorization.k8s.io/cert-manager-cainjector:leaderelection created
    rolebinding.rbac.authorization.k8s.io/cert-manager:leaderelection created
    rolebinding.rbac.authorization.k8s.io/cert-manager-webhook:dynamic-serving created
    service/cert-manager created
    service/cert-manager-webhook created
    deployment.apps/cert-manager-cainjector created
    deployment.apps/cert-manager created
    deployment.apps/cert-manager-webhook created
    mutatingwebhookconfiguration.admissionregistration.k8s.io/cert-manager-webhook created
    validatingwebhookconfiguration.admissionregistration.k8s.io/cert-manager-webhook created

.. raw:: html

   </details>

Check whether all cert-manager pods are running.

.. code:: bash

    $ kubectl get pod -n cert-manager

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME                                       READY   STATUS    RESTARTS   AGE
    cert-manager-5dd59d9d9b-cwp8n              1/1     Running   0          58s
    cert-manager-cainjector-8696fc9f89-tvftj   1/1     Running   0          58s
    cert-manager-webhook-7d4b5b8c56-htchs      1/1     Running   0          58s

.. raw:: html

   </details>

Vineyard CRDs、Controllers、Webhooks and Scheduler are packaged by `helm`_, you could 
deploy all resources as follows.

.. code:: bash

    $ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    "vineyard" has been added to your repositories

.. raw:: html

   </details>

Update the vineyard operator chart to the newest one.

.. code:: bash

    $ helm repo update

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    Hang tight while we grab the latest from your chart repositories...
    ...Successfully got an update from the "vineyard" chart repository
    Update Complete. ⎈Happy Helming!⎈

.. raw:: html

   </details>

Deploy the vineyard operator 0.11.4 in the namespace ``vineyard-system``.

.. code:: bash

    $ helm install vineyard-operator vineyard/vineyard-operator -n vineyard-system --version 0.11.4 

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME: vineyard-operator
    LAST DEPLOYED: Wed Jan  4 16:41:45 2023
    NAMESPACE: vineyard-system
    STATUS: deployed
    REVISION: 1
    TEST SUITE: None
    NOTES:
    Thanks for installing VINEYARD-OPERATOR:0.11.4, release at namespace: vineyard-system, name: vineyard-operator.

    To learn more about the release, try:

    $ helm status vineyard-operator -n vineyard-system   # get status of running vineyard operator
    $ helm get all vineyard-operator -n vineyard-system  # get all deployment yaml of vineyard operator

    To uninstall the release, try:

    $ helm uninstall vineyard-operator -n vineyard-system

    You could get all details about vineyard operator in the doc [https://v6d.io/notes/vineyard-operator.html], just have fun with vineyard operator!

.. raw:: html

   </details>

Check the status of all vineyard resources created by helm.

.. code:: bash

    $ kubectl get all -n vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME                                   READY   STATUS    RESTARTS   AGE
    pod/vineyard-operator-cbcd58cb-5zs84   2/2     Running   0          4m56s

    NAME                                        TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
    service/vineyard-operator-metrics-service   ClusterIP   10.96.23.137   <none>        8443/TCP   4m56s
    service/vineyard-operator-webhook-service   ClusterIP   10.96.215.18   <none>        443/TCP    4m56s

    NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/vineyard-operator   1/1     1            1           4m56s

    NAME                                         DESIRED   CURRENT   READY   AGE
    replicaset.apps/vineyard-operator-cbcd58cb   1         1         1       4m56s

.. raw:: html

   </details>

3. Deploy a vineyard cluster
----------------------------

Once you have installed the vineyard operator following the steps above, then deploy 
a vineyard cluster with two vineyard instances by creating a ``Vineyardd`` CR as follows. 

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

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    vineyardd.k8s.v6d.io/vineyardd-sample created

.. raw:: html

   </details>

Check the status of all relevant resources managed by the ``vineyardd-sample`` cr.

.. code:: bash

    $ kubectl get all -l app.kubernetes.io/instance=vineyardd -n vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME                                   READY   STATUS    RESTARTS   AGE
    pod/vineyardd-sample-879798cb6-qpvtw   1/1     Running   0          2m59s
    pod/vineyardd-sample-879798cb6-x4m2x   1/1     Running   0          2m59s

    NAME                               READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/vineyardd-sample   2/2     2            2           2m59s

    NAME                                         DESIRED   CURRENT   READY   AGE
    replicaset.apps/vineyardd-sample-879798cb6   2         2         2       2m59s

.. raw:: html

   </details>

4. Connect to vineyard cluster
------------------------------

Vineyard client support C++(mature)、python(mature)、java(immature)、golang(immature) 
and rust(immature) at present. Here for showing the feature conveniently, we use the 
python client as an example to access the vineyard cluster. Besides, vineyard provides 
two connection methods: `IPC and RPC`_. Next we will explain in trun.

Deploy the python client on two vineyard nodes as follows.

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
        annotations:
            scheduling.k8s.v6d.io/required: none
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
            image: vineyardcloudnative/vineyard-python:v0.11.4
            command: 
            - /bin/bash
            - -c
            - sleep infinity
            volumeMounts:
            - mountPath: /var/run
                name: vineyard-sock
        volumes:
        - name: vineyard-sock
            hostPath:
            path: /var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample
    EOF

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    pod/vineyard-python-client created

.. raw:: html

   </details>

Wait for the vineyard python client pod ready.

.. code:: bash

    $ kubectl get pod -l app=vineyard-python-client -n vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    NAME                                      READY   STATUS    RESTARTS   AGE
    vineyard-python-client-6fd8c47c98-7btkv   1/1     Running   0          93s

.. raw:: html

   </details>

Use the kubectl exec command to enter the first vineyard python client pod.

.. code:: bash

    $ kubectl exec -it $(kubectl get pod -l app=vineyard-python-client -n vineyard-system -oname | head -n 1 | awk -F '/' '{print $2}') -n vineyard-system /bin/bash

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
    root@vineyard-python-client-6fd8c47c98-schvh:/#

.. raw:: html

   </details>

Then you can connect to the vineyard cluster by IPC.

.. code-block:: python

    root@vineyard-python-client-6fd8c47c98-schvh:/# ipython3
    Python 3.10.4 (main, May 11 2022, 07:15:55) [GCC 10.2.1 20210110]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 8.8.0 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: import vineyard

    In [2]: import numpy as np

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

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    kubectl exec [POD] [COMMAND] is DEPRECATED and will be removed in a future version. Use kubectl exec [POD] -- [COMMAND] instead.
    root@vineyard-python-client-6fd8c47c98-zz7p7:/#

.. raw:: html

   </details>

Also, you can connect to the vineyard cluster by RPC and get the metadata of 
above object as follows.

.. code-block:: python

    root@vineyard-python-client-6fd8c47c98-zz7p7:/# ipython3
    Python 3.10.4 (main, May 11 2022, 07:15:55) [GCC 10.2.1 20210110]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 8.8.0 -- An enhanced Interactive Python. Type '?' for help.

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

5. Destroy the vineyard operator and kubernetes cluster
-------------------------------------------------------

Destroy the vineyard operator via helm.

.. code:: bash

    $ helm uninstall vineyard-operator -n vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    release "vineyard-operator" uninstalled

.. raw:: html

   </details>

Delete the namespace.

.. code:: bash

    $ kubectl delete namespace vineyard-system

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    namespace "vineyard-system" deleted

.. raw:: html

   </details>

Destory the kubernetes cluster created by kind.

.. code:: bash

    $ kind delete cluster

.. raw:: html

   <details>
   <summary><a>Expeced output</a></summary>

.. code:: bash

    Deleting cluster "kind" ...

.. raw:: html

   </details>

.. _kind: https://kind.sigs.k8s.io
.. _helm: https://helm.sh/docs/intro/install/
.. _IPC and RPC: https://v6d.io/notes/data-accessing.html#ipcclient-vs-rpcclient
.. _vineyard data accessing: https://v6d.io/notes/data-accessing.html