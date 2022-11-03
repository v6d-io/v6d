Vineyard Operator Guide
=======================

To manage all vineyard relevant components in kubernetes cluster, we proposal the vineyard operator.
With it, users can easily deploy vineyard components, and manage their lifecycle of them. The guide will 
show you all details about vineyard operator and how to use it to manage vineyard components.

Table of Contents
-----------------

* `Install <#install>`_

  * `Install from helm chart <#install-from-helm-chart>`_
  * `Install from source code <#install-from-source-code>`_

* `Features of Vineyard Operator <#features-of-vineyard-operator>`_
* `CustomResourceDefinitions <#customresourcedefinitions>`_

  * `Vineyardd <#vineyardd>`_
  * `GlobalObject <#globalobject>`_
  * `LocalObject <#localobject>`_
  * `Operation <#operation>`_

* `Vineyard Scheduler <#vineyard-scheduler>`_
* `Pluggable Drivers <#pluggable-drivers>`_

  * `Checkpoint <#checkpoint>`_
  * `Assembly <#assembly>`_
  * `Repartition <#repartition>`_

Install
-------

There are two ways to install vineyard operator, one is to install it from helm chart(recommended way), and the other is to
install it from source code.

**Note** Before you install vineyard operator, you should have a kubernetes cluster and kubectl installed.
Here we use `kind <https://kind.sigs.k8s.io/>`_ as an example.

Install from helm chart
^^^^^^^^^^^^^^^^^^^^^^^

1. Install cert-manager:

.. code:: bash

    $  kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml

    or 

    $  helm repo add jetstack https://charts.jetstack.io
    $  helm install \
          cert-manager jetstack/cert-manager \
          --namespace cert-manager \
          --create-namespace \
          --version v1.9.1 \
          --set installCRDs=true

**Note** Please wait the cert-manager for a while until it is ready. Then, install the 
vineyard operator.

2. Install vineyard operator:

.. code:: bash

    $  helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
    $  helm install vineyard-operator vineyard/vineyard-operator

3. Check the vineyard opeartor:

.. code:: bash

    $  kubectl get all -n vineyard-system
    NAME                                               READY   STATUS    RESTARTS   AGE
    pod/vineyard-controller-manager-5c6f4bc454-8xm8q   2/2     Running   0          62m

    NAME                                                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
    service/vineyard-controller-manager-metrics-service   ClusterIP   10.96.240.173   <none>        8443/TCP   62m
    service/vineyard-webhook-service                      ClusterIP   10.96.41.132    <none>        443/TCP    62m

    NAME                                          READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/vineyard-controller-manager   1/1     1            1           62m

    NAME                                                     DESIRED   CURRENT   READY   AGE
    replicaset.apps/vineyard-controller-manager-5c6f4bc454   1         1         1       62m


Install from source code
^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the vineyard repo:

.. code:: bash

    $ git clone https://github.com/v6d-io/v6d.git

2. Install cert-manager:

.. code:: bash

    $ kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml

3. Build the vineyard operator image and deploy it to your kubernetes cluster:

.. code:: bash

    $ cd k8s
    $ make -C k8s docker-build
    $ kind load docker-image registry-vpc.cn-hongkong.aliyuncs.com/libvineyard/vineyard-operator:latest
    $ make -C k8s deploy
    $ make -C k8s deploy

4. Check the vineyard operator as below:

.. code:: bash

    $  kubectl get all -n vineyard-system
    NAME                                               READY   STATUS    RESTARTS   AGE
    pod/vineyard-controller-manager-5c6f4bc454-8xm8q   2/2     Running   0          62m

    NAME                                                  TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
    service/vineyard-controller-manager-metrics-service   ClusterIP   10.96.240.173   <none>        8443/TCP   62m
    service/vineyard-webhook-service                      ClusterIP   10.96.41.132    <none>        443/TCP    62m

    NAME                                          READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/vineyard-controller-manager   1/1     1            1           62m

    NAME                                                     DESIRED   CURRENT   READY   AGE
    replicaset.apps/vineyard-controller-manager-5c6f4bc454   1         1         1       62m


Features of Vineyard Operator
-----------------------------

There are three main components in vineyard operator:
- CustomResourceDefinitions(CRDs).
- Vineyard Scheduler.
- Pluggable Drivers.

Next, we will introduce them in detail.


CustomResourceDefinitions
-------------------------

Kubernetes provides the `CustomResouceDefinitions(CRDs) <https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/>`_ 
for users to create their own resources. The vineyard operator is based on CRDs, and it provides the 
following CRDs:

Vineyardd
^^^^^^^^^

The **Vineyardd** custom resource definition (CRD) declaratively defines the vineyard daemon server in a
a kubernetes cluster. It provides the following options to configure the vineyard daemon server:

+-----------------------------------------------+-----------------------------------+----------------------------------------------------------------------+------------------------------------------------+
| Option Name                                   | Type                              | Descrition                                                           | Default Value                                  |
+===============================================+===================================+======================================================================+================================================+
| image                                         | string                            | The image name of vineyardd image                                    | "vineyardcloudnative/vineyardd:alpine-latest"  |
| imagePullPolicy                               | string                            | The image pull policy of vineyardd image                             | nil                                            |
| version                                       | string                            | The version of vineyardd                                             | "latest"                                       |
| replicas                                      | int                               | The replicas of vineyardd                                            | 3                                              |
| env                                           | []corev1.EnvVar                   | The environment of vineyardd                                         | nil                                            |
| metric.image                                  | string                            | The image name of metric                                             | nil                                            |
| metric.imagePullPolicy                        | string                            | The image pull policy of metric                                      | nil                                            |
| config.etcdCmd                                | string                            | The path of etcd command                                             | nil                                            |
| config.etcdEndpoint                           | string                            | The endpoint of etcd                                                 | nil                                            |
| config.etcdPrefix                             | string                            | The path prefix of etcd                                              | nil                                            |
| config.enableMetrics                          | bool                              | Enable the metrics in vineyardd                                      | false                                          |
| config.enablePrometheus                       | bool                              | Enable the Prometheus                                                | false                                          |
| config.socket                                 | string                            | The ipc socket file of vineyardd                                     | nil                                            |
| config.streamThreshold                        | int64                             | The memory threshold of streams (percentage of total memory)         | nil                                            |
| config.sharedMemorySize                       | string                            | The shared memory size for vineyardd                                 | nil                                            |
| config.syncCRDs                               | bool                              | Synchronize CRDs when persisting objects                             | nil                                            |
| config.spillConfig.Name                       | string                            | The name of the spill config, if set we'll enable the spill module.  | nil                                            |
| config.spillConfig.path                       | string                            | The path of spilling                                                 | nil                                            |
| config.spillConfig.spillLowerRate             | string                            | The low watermark of spilling memory                                 | nil                                            |
| config.spillConfig.spillUpperRate             | string                            | The high watermark of triggering spilling                            | nil                                            |
| config.spillConfig.persistentVolumeSpec       | corev1.PersistentVolumeSpec       | The PV of the spilling for persistent storage                        | nil                                            |
| config.spillConfig.persistentVolumeClaimSpec  | corev1.PersistentVolumeClaimSpec  | The PVC of the spilling for the persistent storage                   | nil                                            |
| service.type                                  | string                            | The service type of vineyardd service                                | nil                                            |
| service.port                                  | int                               | The service port of vineyardd service                                | nil                                            |
| etcd.instances                                | int                               | The etcd instances of vineyard                                       | nil                                            |
+-----------------------------------------------+-----------------------------------+----------------------------------------------------------------------+------------------------------------------------+

You could use the following yaml file to create a default vineyardd daemon server with three instances:

.. code:: yaml

    cat <<EOF | kubectl apply -f -
    apiVersion: k8s.v6d.io/v1alpha1
    kind: Vineyardd
    metadata:
      name: vineyardd-sample
      namespace: vineyard-system
    spec:
      image: ghcr.io/v6d-io/v6d/vineyardd:alpine-latest
      version: latest
      replicas: 3
      imagePullPolicy: IfNotPresent
      # vineyardd's configuration
      config:
        syncCRDs: true
        enableMetrics: false
      etcd:
        instances: 3
      service:
        type: ClusterIP
        port: 9600
    EOF

Check all deployments and services:

.. code:: bash

    $ kubectl get all -n vineyard-system
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

GlobalObject
^^^^^^^^^^^^

The **GlobalObject** custom resource definition (CRD) declaratively defines the global object in a kubernetes
cluster, it contains the following fields:

+--------------+-----------+------------------------------------------------------------------+----------------+
| Option Name  | Type      | Descrition                                                       | Default Value  |
+==============+===========+==================================================================+================+
| id           | string    | The id of globalobject                                           | nil            |
| name         | string    | The name of globalobject, the same as id.                        | nil            |
| signature    | string    | The signature of the globalobject                                | nil            |
| typename     | string    | The typename of globalobject, including the vineyard's core type | nil            |
| members      | []string  | The signatures of all localobjects contained in the globalobject | nil            |
| metadata     | string    | The same as typename                                             | nil            |
+--------------+-----------+------------------------------------------------------------------+----------------+

In general, the GlobalObjects are created as intermediate objects when deploying users' applications. You could get them
as follows.

.. code:: bash

    $ kubectl get globalobjects -A
    NAMESPACE         NAME                ID                  NAME   SIGNATURE           TYPENAME
    vineyard-system   o001bcbcea406acd0   o001bcbcea406acd0          s001bcbcea4069f60   vineyard::GlobalDataFrame
    vineyard-system   o001bcc19dbfc9c34   o001bcc19dbfc9c34          s001bcc19dbfc8d7a   vineyard::GlobalDataFrame

LocalObject
^^^^^^^^^^^

The **LocalObject** custom resource definition (CRD) declaratively defines the local object in a kubernetes
cluster, it contains the following fields:

+--------------+---------+-------------------------------------------------------------------+----------------+
| Option Name  | Type    | Descrition                                                        | Default Value  |
+==============+=========+===================================================================+================+
| id           | string  | The id of localobject                                             | nil            |
| name         | string  | The name of localobject, the same as id.                          | nil            |
| signature    | string  | The signature of localobjects                                     | nil            |
| typename     | string  | The typename of localobjects, including the vineyard's core type  | nil            |
| instance_id  | int     | The instance id created by vineyard daemon server                 | nil            |
| hostname     | string  | The hostname of localobjects locations                            | nil            |
| metadata     | string  | The same as typename                                              | nil            |
+--------------+---------+-------------------------------------------------------------------+----------------+

The LocalObjects are also intermediate objects just like the GlobalObjects, and you could get them as follows.

.. code:: bash

    $ kubectl get localobjects -A
    NAMESPACE         NAME                ID                  NAME   SIGNATURE           TYPENAME              INSTANCE   HOSTNAME
    vineyard-system   o001bcbce202ab390   o001bcbce202ab390          s001bcbce202aa6f6   vineyard::DataFrame   0          kind-worker2
    vineyard-system   o001bcbce21d273e4   o001bcbce21d273e4          s001bcbce21d269c2   vineyard::DataFrame   1          kind-worker
    vineyard-system   o001bcbce24606f6a   o001bcbce24606f6a          s001bcbce246067fc   vineyard::DataFrame   2          kind-worker3

Operation
^^^^^^^^^

The **Operation** custom resource definition (CRD) declaratively defines the configurable pluggable drivers 
( mainly `assembly` and `repartition` ) in a kubernetes cluster, it contains the following fields:

+-----------------+---------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------+
| Option Name     | Type    | Description                                                                                                                                                                              | Default Value  |
+=================+=========+==========================================================================================================================================================================================+================+
| name            | string  | the name of vineyard pluggable drivers, including assembly and repartition.                                                                                                              | nil            |
| type            | string  | the type of operation. For `assembly`, it mainly contains `local (for localobject)` and `distributed (for globalobject)`. For `repartition`, it contains `dask (object built in dask)`.  | nil            |
| require         | string  | The required job's name of the operation                                                                                                                                                 | nil            |
| target          | string  | The target job's name of the operation                                                                                                                                                   | nil            |
| timeoutSeconds  | string  | The timeout of the operation.                                                                                                                                                            | 300            |
+-----------------+---------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------+

The operation CR is created by the vineyard scheduler while scheduling the vineyard jobs, and you could get them as follows.

.. code:: bash

    $  kubectl get operation -A
    NAMESPACE      NAME                                    OPERATION     TYPE   STATE
    vineyard-job   dask-repartition-job2-bbf596bf4-985vc   repartition   dask   

Vineyard Scheduler
------------------

The vineyard scheduler is developed based on kubernetes scheduler framework. It is responsible for
scheduling the workload on vineyard. The overall scheduling strategy is as follows.

- All vineyard workloads can only be deployed in the nodes that exists vineyard daemon server.
- If a workload doesn't depend on any other workload, it will be scheduled by **round-robin**. E.g.
  If a workload has 3 replicas and there are 3 nodes that exists vineyard daemon server, the first 
  replica will be scheduled on the first node, the second replica will be scheduled on the second node, 
  and so on.
- If a workload depends on other workloads, it will be scheduled by **best-effort**. Assuming a workload
  produces N chunks during its lifecycle, and there are M nodes that exists vineyard daemon server, the
  best-effort policy will try to make the next workload consume M/N chunks. E.g. Image a workload produces
  12 chunks and their distributions are as follows:
  .. code:: yaml

    node0: 0-8
    node1: 9-11
    node2: 12

  The next workload has 3 replicas, and the best-effort policy will schedule it as follows:

  .. code:: yaml

    replica1 -> node1 (consume 0-3 chunks)
    replica2 -> node1 (consume 4-7 chunks)
    replica3 -> node2 (consume 9-11 chunks, the other chunks will be migrated to the node)

How to use the vineyard scheduler?
""""""""""""""""""""""""""""""""""

To make the deployment easier, we intergrate the vineyard scheduler into the vineyard operator, so when users
deploy the vineyard operator, we can use the scheduler at the same time. The following tables show all 
configurations you have to set up first.

+---------------------------------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Name                            | Yaml fields  | Description                                                                                                                                                                                   |
+=================================+==============+===============================================================================================================================================================================================+
| scheduling.k8s.v6d.io/required  | annotations  | All jobs required by the job. If there are more than two tasks, use the concatenator '.' to concatenate them into a string. E.g. `job1.job2.job3`. If there is no required jobs, set `none`.  |
| scheduling.k8s.v6d.io/vineyardd | labels       | The name of vineyardd. Notice, the vineyardd's namespace is generally recommended to be set to `vineyard-system`.                                                    |
| scheduling.k8s.v6d.io/job       | labels       | The job name.                                                                                                                                                                                 |
| schdulerNmae                    | spec         | The vineyard sheduler's name, and the default value is `vineyard-scheduler`.                                                                                                                  |
+---------------------------------+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Next, we will show a complete example of how to use the vineyard scheduler. First, we should install the 
vineyard operator and vineyard daemon server following the previous steps, then deploy `a workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/workflow-demo/job1.py>`_ as follows.

.. code:: bash
    $ kubectl create ns vineyard-job
    $ cat <<EOF | kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: v6d-workflow-demo-job1-deployment
      namespace: vineyard-job
    spec:
    selector:
      matchLabels:
        app: v6d-workflow-demo-job1
    replicas: 2
    template:
      metadata:
        annotations:
          # required jobs
          scheduling.k8s.v6d.io/required: none
        labels:
          app: v6d-workflow-demo-job1
          # vineyardd's name
          scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
          # job name
          scheduling.k8s.v6d.io/job: v6d-workflow-demo-job1
      spec:
        # vineyard scheduler name
        schedulerName: vineyard-scheduler
        containers:
        - name: job1
          image: ghcr.io/v6d-io/v6d/job1
          # please set the JOB_NAME env, it will be used by vineyard scheduler
          env:
          - name: JOB_NAME
            value: v6d-workflow-demo-job1
          imagePullPolicy: IfNotPresent
          volumeMounts:
          - mountPath: /var/run
            name: vineyard-sock
        volumes:
        - name: vineyard-sock
          hostPath:
            # the path name is combined by `vineyard-{vineyardd's namespace}-{vineyardd's name}`
            path: /var/run/vineyard-vineyard-system-vineyardd-sample
    EOF

Check the job and the objects produced by it:

.. code:: bash

    $ kubectl get all -n vineyard-job
    NAME                                                     READY   STATUS    RESTARTS   AGE
    pod/v6d-workflow-demo-job1-deployment-6f479d695b-698xb   1/1     Running   0          3m16s
    pod/v6d-workflow-demo-job1-deployment-6f479d695b-7zrw6   1/1     Running   0          3m16s

    NAME                                                READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/v6d-workflow-demo-job1-deployment   2/2     2            2           3m16s

    NAME                                                           DESIRED   CURRENT   READY   AGE
    replicaset.apps/v6d-workflow-demo-job1-deployment-6f479d695b   2         2         2       3m16s

    $ kubectl get globalobjects -n vineyard-system
    NAME                ID                  NAME   SIGNATURE           TYPENAME
    o001c87014cf03c70   o001c87014cf03c70          s001c87014cf03262   vineyard::Sequence
    o001c8729e49e06b8   o001c8729e49e06b8          s001c8729e49dfbb4   vineyard::Sequence

    $ kubectl get localobjects -n vineyard-system
    NAME                ID                  NAME   SIGNATURE           TYPENAME                  INSTANCE   HOSTNAME
    o001c87014ca81924   o001c87014ca81924          s001c87014ca80acc   vineyard::Tensor<int64>   1          kind-worker2
    o001c8729e4590626   o001c8729e4590626          s001c8729e458f47a   vineyard::Tensor<int64>   2          kind-worker3

    # when a job is scheduled, the scheduler will create a configmap to record the globalobject id 
    # that the next job will consume.
    $ kubectl get configmap v6d-workflow-demo-job1 -n vineyard-job -oyaml
    apiVersion: v1
    data:
      kind-worker3: o001c8729e4590626
      v6d-workflow-demo-job1: o001c8729e49e06b8
    kind: ConfigMap
    ...

Then deploy the `second workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/workflow-demo/job2.py>`_ as follows.

.. code:: bash

    $ cat <<EOF | kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: v6d-workflow-demo-job2-deployment
      namespace: vineyard-job
    spec:
      selector:
        matchLabels:
          app: v6d-workflow-demo-job2
    replicas: 3
    template:
      metadata:
        annotations:
          # required jobs
          scheduling.k8s.v6d.io/required: v6d-workflow-demo-job1
        labels:
          app: v6d-workflow-demo-job2
          # vineyardd's name
          scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
          # job name
          scheduling.k8s.v6d.io/job: v6d-workflow-demo-job2
        spec:
          # vineyard scheduler name
          schedulerName: vineyard-scheduler
          containers:
          - name: job2
            image: ghcr.io/v6d-io/v6d/job2
            imagePullPolicy: IfNotPresent
            env:
            - name: JOB_NAME
              value: v6d-workflow-demo-job2
            # pass node name to the environment
            - name: NODENAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            # Notice, vineyard operator will create a configmap to pass the global object id produced by the previous job.
            # Please set the configMapRef, it's name is the same as the job name.
            envFrom:
              - configMapRef:
                  name: v6d-workflow-demo-job1
            volumeMounts:
            - mountPath: /var/run
              name: vineyard-sock
          volumes:
          - name: vineyard-sock
            hostPath:
              path: /var/run/vineyard-vineyard-system-vineyardd-sample
    EOF

Check if all jobs are ready:

.. code:: bash

    $ kubectl get all -n vineyard-job             
    NAME                                                     READY   STATUS    RESTARTS      AGE
    pod/v6d-workflow-demo-job1-deployment-6f479d695b-698xb   1/1     Running   0             8m12s
    pod/v6d-workflow-demo-job1-deployment-6f479d695b-7zrw6   1/1     Running   0             8m12s
    pod/v6d-workflow-demo-job2-deployment-b5b58cbdc-4s7b2    1/1     Running   0             6m24s
    pod/v6d-workflow-demo-job2-deployment-b5b58cbdc-cd5v2    1/1     Running   0             6m24s
    pod/v6d-workflow-demo-job2-deployment-b5b58cbdc-n6zvm    1/1     Running   0             6m24s

    NAME                                                READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/v6d-workflow-demo-job1-deployment   2/2     2            2           8m12s
    deployment.apps/v6d-workflow-demo-job2-deployment   3/3     3            3           6m24s

    NAME                                                           DESIRED   CURRENT   READY   AGE
    replicaset.apps/v6d-workflow-demo-job1-deployment-6f479d695b   2         2         2       8m12s
    replicaset.apps/v6d-workflow-demo-job2-deployment-b5b58cbdc    3         3         3       6m24s

The above is the process of running the workload based on the vineyard shceduler, and it's same as the `e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/vineyardd/e2e.yaml>`_. 
What's more, you could refer to the `workflow demo <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/workflow-demo>`_  to dig what happens in the container. 

Pluggable Drivers
-----------------

At present, vineyard operator contains three pluggable drivers: `checkpoint`, `assembly`, and `repartition`. Next is a brief 
introduction of them.

Checkpoint
^^^^^^^^^^

Now there are two kinds of checkpoint drivers in vineyard.

1. Active checkpoint - **Serialization**. Users can store data in temporary/persistent storage for checkpoint by the API
(`vineyard.io.serialize/deserialize`). *Notice*, the serialization is triggered by the user in the application image, and the 
volume is also created by the user, so it's not managed by the vineyard operator.

2. Passive checkpoint - **Spill**. Now vineyard supports spilling data from memory to storage while the data is too large to be stored. 
There are two watermarks of spilling memory, the low watermark and the high watermark. When the data is larger than the high watermark, 
vineyardd will spill the extra data to the storage until it is at the low watermark.

How to use checkpoint in vineyard operator?
"""""""""""""""""""""""""""""""""""""""""""

Now, the checkpoint driver(**Spill**) is configured in the `vineyardd` custom resource definition (CRD). You could use the following yaml 
file to create a default vineyardd daemon server with spill mechanism:

.. note::

    The spill mechanism supports temporary storage (`HostPath <https://kubernetes.io/docs/concepts/storage/volumes/#hostpath>`_)
    and persistent storage (`PersistentVolume <https://kubernetes.io/docs/concepts/storage/persistent-volumes/>`_)

.. code:: bash

    $ cat <<EOF | kubectl apply -f -
    apiVersion: k8s.v6d.io/v1alpha1
    kind: Vineyardd
    metadata:
      name: vineyardd-sample
      namespace: vineyard-system
    spec:
      image: ghcr.io/v6d-io/v6d/vineyardd:alpine-latest
      version: latest
      replicas: 3
      imagePullPolicy: IfNotPresent
      # vineyardd's configuration
      config:
        sharedMemorySize: "2048"
        syncCRDs: true
        enableMetrics: false
        # spill configuration
        spillConfig:
          # if set, then enable the spill mechanism
          name: spill-path
          # please make sure the path exists
          path: /var/vineyard/spill
          spillLowerRate: "0.3"
          spillUpperRate: "0.8"
          persistentVolumeSpec:
            storageClassName: manual
            capacity:
              storage: 1Gi
            accessModes:
              - ReadWriteOnce
            hostPath:
              path: /var/vineyard/spill
        persistentVolumeClaimSpec:
            storageClassName: manual
            accessModes:
              - ReadWriteOnce
            resources:
              requests:
                storage: 512Mi
        etcd:
          instances: 3
        service:
          type: ClusterIP
          port: 9600
    EOF

For more information about the checkpoint mechanism in vineyard operator, there are `some examples <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/checkpoint-demo>`_.
Besides, you could refer to `the serialize e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/serialize/e2e.yaml>`_ and 
`the spill e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/spill/e2e.yaml>`_ to get some inspiration on how to use the 
checkpoint mechanism in a workflow.

Assembly
^^^^^^^^

In actual usage scenarios, there are different kinds of computing engines in a workload. Some computing engines may support the stream 
types to speed up data processing, while some computing engines don't support the stream types. To make the workload work as expected, 
we need to add an assembly mechanism to transform the steam type to the chunk type so that the next computing engine which can't use 
the stream type could read the metadata produced by the previous engine.

How to use assembly in vineyard operator?
"""""""""""""""""""""""""""""""""""""""""

For reducing the stress of Kubernetes API Server, we provide the namespace selector for assembly. The assembly driver will only be applyed
in the namespace with the specific label `operation-injection: enabled`. Therefore, please make sure the applications' namespace has the
label before using the assembly mechanism.

We provide some labels to help users to use the assembly mechanism in vineyard operator. The following is all labels
that we provide:

+--------------------------+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Name                     | Yaml fields  | Description                                                                                                                                                  |
+==========================+==============+==============================================================================================================================================================+
| assembly.v6d.io/enabled  | labels       | If the job needs an assembly operation before deploying it, then set `true`                                                                                  |
| assembly.v6d.io/type     | labels       | There are two types in assembly operation, `local` only for localobject(stream on the same node), `distributed` for globalobject(stream on different nodes)  |
+--------------------------+--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

Next, we will show how to use the assembly mechanism in vineyard operator. Assuming that we have a workflow that contains two workloads, 
the first workload is a stream workload and the second workload is a chunk workload. The following is the yaml file of the `first workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/assembly-demo/assembly-job1.py>`_:

**Note** Please make sure you have installed the vineyard operator and vineyardd before running the following yaml file.

.. code:: bash
    $ kubectl create namespace vineyard-job
    $ cat <<EOF | kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: assembly-job1
      namespace: vineyard-job
    spec:
      selector:
        matchLabels:
          app: assembly-job1
      replicas: 1
      template:
        metadata:
          annotations:
            scheduling.k8s.v6d.io/required: none
          labels:
            app: assembly-job1
            # this label represents the vineyardd's name that need to be used
            scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
            scheduling.k8s.v6d.io/job: assembly-job1
        spec:
          schedulerName: vineyard-scheduler
          containers:
            - name: assembly-job1
              image: ghcr.io/v6d-io/v6d/assembly-job1
              env:
                - name: JOB_NAME
                  value: assembly-job1
              imagePullPolicy: IfNotPresent
              volumeMounts:
                - mountPath: /var/run
                  name: vineyard-sock
          volumes:
            - name: vineyard-sock
              hostPath:
                path: /var/run/vineyard-vineyard-system-vineyardd-sample
    EOF
    # we can get the localobjects produced by the first workload, it's a stream type.
    $ kubectl get localobjects -n vineyard-system
    NAME                ID                  NAME   SIGNATURE           TYPENAME                      INSTANCE   HOSTNAME
    o001d1b280049b146   o001d1b280049b146          s001d1b280049a4d4   vineyard::RecordBatchStream   0          kind-worker2

From the above output, we can see that the localobjects produced by the first workload is a stream type. Next, we deploy the second 
workload with the assembly mechanism. The following is the yaml file of the `second workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/assembly-demo/assembly-job2.py>`_:

.. code:: bash

  # remember label the namespace with the label `operation-injection: enabled` to enable pluggable drivers.
  $ kubectl label namespace vineyard-job operation-injection=enabled
  $ cat <<EOF | kubectl apply -f -
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: assembly-job2
    namespace: vineyard-job
  spec:
    selector:
      matchLabels:
        app: assembly-job2
    replicas: 1
    template:
      metadata:
        annotations:
          scheduling.k8s.v6d.io/required: assembly-job1
        labels:
          app: assembly-job2
          assembly.v6d.io/enabled: "true"
          assembly.v6d.io/type: "local"
          # this label represents the vineyardd's name that need to be used
          scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
          scheduling.k8s.v6d.io/job: assembly-job2
      spec:
        schedulerName: vineyard-scheduler
        containers:
          - name: assembly-job2
            image: ghcr.io/v6d-io/v6d/assembly-job2
            env:
              - name: JOB_NAME
                value: assembly-job2
              - name: REQUIRED_JOB_NAME
                value: assembly-job1
            envFrom:
            - configMapRef:
                name: assembly-job1
            imagePullPolicy: IfNotPresent
            volumeMounts:
              - mountPath: /var/run
                name: vineyard-sock
        volumes:
          - name: vineyard-sock
            hostPath:
              path: /var/run/vineyard-vineyard-system-vineyardd-sample
  EOF


After the second workload is deployed, it is still pending, which means that the scheduler recognizes 
that the workload needs an assembly operation, so the following assembly operation CR will be created.

.. code:: bash

  # get all workloads, the job2 is pending as it needs an assembly operation.
  $ kubectl get pod -n vineyard-job
  NAME                             READY   STATUS    RESTARTS   AGE
  assembly-job1-86c99c995f-nzns8   1/1     Running   0          2m
  assembly-job2-646b78f494-cvz2w   0/1     Pending   0          53s

  # the assembly operation CR is created by the scheduler.
  $ kubectl get operation -A
  NAMESPACE      NAME                             OPERATION   TYPE    STATE
  vineyard-job   assembly-job2-646b78f494-cvz2w   assembly    local   

During the assembly operation, the Operation Controller will create a job to run assembly operation. We can get the objects produced by 
the job.

.. code:: bash

  # get the assembly operation job
  $ kubectl get job -n vineyard-job
  NAMESPACE      NAME                         COMPLETIONS   DURATION   AGE
  vineyard-job   assemble-o001d1b280049b146   1/1           26s        119s
  # get the pod
  $ kubectl get pod -n vineyard-job                                     
  NAME                               READY   STATUS      RESTARTS   AGE
  assemble-o001d1b280049b146-fzws7   0/1     Completed   0          5m55s
  assembly-job1-86c99c995f-nzns8     1/1     Running     0          4m
  assembly-job2-646b78f494-cvz2w     0/1     Pending     0          5m
  
  # get the localobjects produced by the job
  $ kubectl get localobjects -l k8s.v6d.io/created-podname=assemble-o001d1b280049b146-fzws7 -n vineyard-system
  NAME                ID                  NAME   SIGNATURE           TYPENAME              INSTANCE   HOSTNAME
  o001d1b56f0ec01f8   o001d1b56f0ec01f8          s001d1b56f0ebf578   vineyard::DataFrame   0          kind-worker2
  o001d1b5707c74e62   o001d1b5707c74e62          s001d1b5707c742e0   vineyard::DataFrame   0          kind-worker2
  o001d1b571f47cfe2   o001d1b571f47cfe2          s001d1b571f47c3c0   vineyard::DataFrame   0          kind-worker2
  o001d1b5736a6fd6c   o001d1b5736a6fd6c          s001d1b5736a6f1cc   vineyard::DataFrame   0          kind-worker2
  o001d1b574d9b94ae   o001d1b574d9b94ae          s001d1b574d9b8a9e   vineyard::DataFrame   0          kind-worker2
  o001d1b5765629cbc   o001d1b5765629cbc          s001d1b57656290a8   vineyard::DataFrame   0          kind-worker2
  o001d1b57809911ce   o001d1b57809911ce          s001d1b57809904e0   vineyard::DataFrame   0          kind-worker2
  o001d1b5797a9f958   o001d1b5797a9f958          s001d1b5797a9ee82   vineyard::DataFrame   0          kind-worker2
  o001d1b57add9581c   o001d1b57add9581c          s001d1b57add94e62   vineyard::DataFrame   0          kind-worker2
  o001d1b57c53875ae   o001d1b57c53875ae          s001d1b57c5386a22   vineyard::DataFrame   0          kind-worker2
  
  # get the globalobjects produced by the job
  $ kubectl get globalobjects -l k8s.v6d.io/created-podname=assemble-o001d1b280049b146-fzws7 -n vineyard-system
  NAME                ID                  NAME   SIGNATURE           TYPENAME
  o001d1b57dc2c74ee   o001d1b57dc2c74ee          s001d1b57dc2c6a4a   vineyard::Sequence


Each stream will be transformed to a globalobject. To make the second workload obtain the globalobject generated by the assembly 
operation, the vineyard scheduler will create a configmap to store the globalobject id as follows.

.. code:: bash

  $ kubectl get configmap assembly-job1 -n vineyard-job -oyaml
  apiVersion: v1
  data:
    assembly-job1: o001d1b57dc2c74ee
  kind: ConfigMap
  ...

When the assembly operation is completed, the scheduler will rescheduler the second workload and it will be 
deployed successfully as follows.

.. code:: bash

  $ kubectl get pod -n vineyard-job                                     
  NAME                               READY   STATUS      RESTARTS   AGE
  assemble-o001d1b280049b146-fzws7   0/1     Completed   0          9m55s
  assembly-job1-86c99c995f-nzns8     1/1     Punning     0          8m
  assembly-job2-646b78f494-cvz2w     1/1     Punning     0          9m

The above process of the assembly operation is shown in the `local assembly e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/assembly/local-assembly-e2e.yaml>`_.
You could refer `assembly workload <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/assembly-demo>`_ and `local assembly operation <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/local-assembly-container>`_ 
to get more details.

Besides, we also support `distributed assembly operation <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/distributed-assembly-container>`_, you could
try the `distributed assembly e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/assembly/distributed-assembly-e2e.yaml>` to get some inspiration.

Repartition
^^^^^^^^^^^

Repartition is a mechanism to repartition the data in the vineyard cluster. It's useful when the number of workers can't meet the need for 
partitions. E.g. Assuming a workload creates 4 partitions, but the number of workers in the next workload is only 3, the repartition mechanism 
will repartition the partitions from 4 to 3 so that the next workload can work as expected. At present, the vineyard operator only supports
repartition based on `dask <https://www.dask.org/get-started>`_.


Dask Repartition Demo
"""""""""""""""""""""

For the workloads based on dask, we provide some annotations and labels to help users to use the assembly mechanism in vineyard operator. The following are all labels
and annotations that we provide:

+---------------------------------------------+--------------+------------------------------------------------------------+
| Name                                        | Yaml fields  | Description                                                |
+=============================================+==============+============================================================+
| scheduling.k8s.v6d.io/dask-scheduler        | annotations  | The service of dask scheduler.                             |
| scheduling.k8s.v6d.io/dask-worker-selector  | annotations  | The label selector of dask worker pod.                     |
| repartition.v6d.io/enabled                  | labels       | Enable the repartition                                     |
| repartition.v6d.io/type                     | labels       | The type of repartition, at present, only support `dask`.  |
| scheduling.k8s.v6d.io/replicas              | labels       | The replicas of the workload                               |
+---------------------------------------------+--------------+------------------------------------------------------------+

The following is a demo of repartition based on dask. At first, we create a dask cluster with 3 workers.

**Note** Please make sure you have installed the vineyard operator and vineyardd before running the following yaml file.

.. code:: bash

  # install dask scheduler and dask worker.
  $ helm repo add dask https://helm.dask.org/
  $ helm repo update
  # the dask-worker's image is built with vineyard, please refer `dask-worker-with-vineyard <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/repartition/repartition-demo/Dockerfile.dask-worker-with-vineyard>`_.
  $ cat <<EOF | helm install dask-cluster dask/dask --values -
  scheduler:
    image:
      tag: "2022.8.1"

  jupyter:
    enabled: false

  worker:
    # worker numbers
    replicas: 3
    image:
      repository: ghcr.io/v6d-io/v6d/dask-worker-with-vineyard
      tag: latest
    env:
      - name: VINEYARD_IPC_SOCKET
        value: /var/run/vineyard.sock
      - name: VINEYARD_RPC_SOCKET
        value: vineyardd-sample-rpc.vineyard-system:9600
    mounts:
      volumes:
        - name: vineyard-sock
          hostPath:
            path: /var/run/vineyard-vineyard-system-vineyardd-sample
      volumeMounts:
        - mountPath: /var/run
          name: vineyard-sock
  EOF

Deploy the `first workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/repartition/repartition-demo/job1.py>`_ as follows:

.. code:: bash

  $ kubectl create namespace vineyard-job
  $ cat <<EOF | kubectl apply -f -
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: dask-repartition-job1
    namespace: vineyard-job
  spec:
    selector:
      matchLabels:
        app: dask-repartition-job1
    replicas: 1
    template:
      metadata:
        annotations:
          scheduling.k8s.v6d.io/required: "none"
          scheduling.k8s.v6d.io/dask-scheduler: "tcp://my-release-dask-scheduler.default:8786"
          # use ',' to separate the different labels here
          scheduling.k8s.v6d.io/dask-worker-selector: "app:dask,component:worker"
        labels:
          app: dask-repartition-job1
          repartition.v6d.io/type: "dask"
          scheduling.k8s.v6d.io/replicas: "1"
          # this label represents the vineyardd's name that need to be used
          scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
          scheduling.k8s.v6d.io/job: dask-repartition-job1
      spec:
        schedulerName: vineyard-scheduler
        containers:
        - name: dask-repartition-job1
          image: ghcr.io/v6d-io/v6d/dask-repartition-job1
          imagePullPolicy: IfNotPresent
          env:
          - name: JOB_NAME
            value: dask-repartition-job1
          - name: DASK_SCHEDULER
            value: tcp://my-release-dask-scheduler.default:8786
          volumeMounts:
          - mountPath: /var/run
            name: vineyard-sock
        volumes:
        - name: vineyard-sock
          hostPath:
            path: /var/run/vineyard-vineyard-system-vineyardd-sample
  EOF

The first workload will create 4 partitions (each partition as a localobject):

.. code:: bash

  $ kubectl get globalobjects -n vineyard-system                                                                
  NAME                ID                  NAME   SIGNATURE           TYPENAME
  o001d2a6ae6c6e2e8   o001d2a6ae6c6e2e8          s001d2a6ae6c6d4f4   vineyard::GlobalDataFrame
  $ kubectl get localobjects -n vineyard-system                                                                
  NAME                ID                  NAME   SIGNATURE           TYPENAME              INSTANCE   HOSTNAME
  o001d2a6a6483ac44   o001d2a6a6483ac44          s001d2a6a6483a3ce   vineyard::DataFrame   1          kind-worker3
  o001d2a6a64a29cf4   o001d2a6a64a29cf4          s001d2a6a64a28f2e   vineyard::DataFrame   0          kind-worker
  o001d2a6a66709f20   o001d2a6a66709f20          s001d2a6a667092a2   vineyard::DataFrame   2          kind-worker2
  o001d2a6ace0f6e30   o001d2a6ace0f6e30          s001d2a6ace0f65b8   vineyard::DataFrame   2          kind-worker2

Deploy the `second workload <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/repartition/repartition-demo/job2.py>`_ as follows:

.. code:: bash

  $ kubectl label namepsace vineyard-job operation-injection=enabled
  $ cat <<EOF | kubectl apply -f -
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: dask-repartition-job2
    namespace: vineyard-job
  spec:
    selector:
      matchLabels:
        app: dask-repartition-job2
    replicas: 1
    template:
      metadata:
        annotations:
          scheduling.k8s.v6d.io/required: "dask-repartition-job1"
          scheduling.k8s.v6d.io/dask-scheduler: "tcp://my-release-dask-scheduler.default:8786"
          # use ',' to separate the different labels here
          scheduling.k8s.v6d.io/dask-worker-selector: "app:dask,component:worker"
        labels:
          app: dask-repartition-job2
          repartition.v6d.io/enabled: "true"
          repartition.v6d.io/type: "dask"
          scheduling.k8s.v6d.io/replicas: "1"
          # this label represents the vineyardd's name that need to be used
          scheduling.k8s.v6d.io/vineyardd: vineyardd-sample
          scheduling.k8s.v6d.io/job: dask-repartition-job2
      spec:
        schedulerName: vineyard-scheduler
        containers:
        - name: dask-repartition-job2
          image: ghcr.io/v6d-io/v6d/dask-repartition-job2
          imagePullPolicy: IfNotPresent
          env:
          - name: JOB_NAME
            value: dask-repartition-job2
          - name: DASK_SCHEDULER
            value: tcp://my-release-dask-scheduler.default:8786
          - name: REQUIRED_JOB_NAME
            value: dask-repartition-job1
          envFrom:
          - configMapRef:
              name: dask-repartition-job1
          volumeMounts:
          - mountPath: /var/run
            name: vineyard-sock
        volumes:
        - name: vineyard-sock
          hostPath:
            path: /var/run/vineyard-vineyard-system-vineyardd-sample
  EOF

The second workload waits for the repartition operation to finish:

.. code:: bash

  # check all workloads
  $ kubectl get pod -n vineyard-job 
  NAME                                     READY   STATUS    RESTARTS   AGE
  dask-repartition-job1-5dbfc54997-7kghk   1/1     Running   0          92s
  dask-repartition-job2-bbf596bf4-cvrt2    0/1     Pending   0          49s

  # check the repartition operation
  $ kubectl get operation -A        
  NAMESPACE      NAME                                    OPERATION     TYPE   STATE
  vineyard-job   dask-repartition-job2-bbf596bf4-cvrt2   repartition   dask   

  # check the job
  $ kubectl get job -n vineyard-job
  NAME                            COMPLETIONS   DURATION   AGE
  repartition-o001d2a6ae6c6e2e8   0/1           8s         8s

After the repartition job finishes, the second workload will be scheduled:

.. code:: bash

  # check all workloads
  kubectl get pod -n vineyard-job
  NAME                                     READY   STATUS      RESTARTS   AGE
  dask-repartition-job1-5dbfc54997-7kghk   1/1     Running     0          5m43s
  dask-repartition-job2-bbf596bf4-cvrt2    0/1     Pending     0          4m30s
  repartition-o001d2a6ae6c6e2e8-79wcm      0/1     Completed   0          3m30s

  # check the repartition operation
  # as the second workload only has 1 replica, the repartition operation will repartitied the global object into 1 partition
  $ kubectl get globalobjects -n vineyard-system
  NAME                ID                  NAME   SIGNATURE           TYPENAME
  o001d2ab523e3fbd0   o001d2ab523e3fbd0          s001d2ab523e3f0e6   vineyard::GlobalDataFrame 

  # the repartition operation will create a new local object(only 1 partition)
  $ kubectl get localobjects -n vineyard-system                                       
  NAMESPACE         NAME                ID                  NAME   SIGNATURE           TYPENAME              INSTANCE   HOSTNAME
  vineyard-system   o001d2dc18d72a47e   o001d2dc18d72a47e          s001d2dc18d729ab6   vineyard::DataFrame   2          kind-worker2

The whole workflow can be found in `dask repartition e2e test <https://github.com/v6d-io/v6d/blob/main/k8s/test/e2e/repartition/dask-repartition-e2e.yaml>`_.
What's more, please refer the `repartition directory <https://github.com/v6d-io/v6d/tree/main/k8s/test/e2e/repartition>`_ to get more details.

Other Repartition Demo
""""""""""""""""""""""

To be done.

End
---

Just have a try, and you will find it easy to use. If you have any questions, 
please feel free to contact us.
