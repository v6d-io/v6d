Frequently Asked Questions
==========================

This *FAQ* page compiles questions frequently asked by our end users to provide
informative and concise answers. If the following sections do not address your
concerns, please feel free to `open an issue`_ or `post it to discussions`_.

1. *What are the objects in vineyard?*

  A global object is composed of multiple local objects distributed across the cluster,
  with each local object stored in a single vineyard daemon (ensuring that a local object
  can always fit into the memory of a single machine).

  These local objects represent partitions of the global object (e.g., partitioned dataframes
  within a large dataframe, graph fragments within a vast graph). Generally, a global object
  serves as an abstraction for the input or output of a parallel-processing workload, while
  a local object corresponds to the input or output of an individual worker within that workload.

2. *Can multiple readers access the same data simultaneously in vineyard?*

  Absolutely. Vineyard stores objects as **immutable** entities, which are shared
  among readers' processes through memory mapping. This ensures safe and concurrent
  access to objects by multiple readers without any conflicts.

3. *How can I launch a cluster with multiple vineyardd instances?*

  A vineyard daemon server represents a single vineyard instance within a vineyard cluster. To
  initiate a vineyard cluster, simply start the ``vineyardd`` process on all the
  machines within the cluster, ensuring that these vineyard instances can register with
  the same ``etcd_endpoint``. The default value for ``etcd_endpoint`` is
  ``http://127.0.0.1:2379``, and if the etcd servers are not already running on the cluster,
  ``vineyard`` will automatically launch the ``etcd_endpoint``.

  For additional parameter settings, refer to the help documentation by running
  ``python3 -m vineyard --help``.

4. *Is Kubernetes a necessity for vineyard?*

  No, Kubernetes is not a necessity for vineyard. However, deploying vineyard on Kubernetes
  allows users to benefit from the flexible resource management offered by cloud-native
  deployments for their application workloads. Additionally, the scheduler plugin assists
  in co-locating worker pods with the data for improved data-work alignment.

5. *How does vineyard achieve IPC and memory sharing (i.e., zero-copy sharing) on Kubernetes?*

  Inter-process memory sharing can be challenging in Kubernetes, but it is achievable. When
  deployed on Kubernetes, vineyard exposes its UNIX-domain socket as a :code:`PersistentVolume`.
  This volume can be mounted into the job's pod, allowing the socket to be used for IPC
  connections to the vineyard daemon. Memory sharing is accomplished by mounting a volume of
  medium :code:`Memory` into both the vineyard daemon's pod and the job's pod.

6. *How does vineyard's stream differ from similar systems, such as Kafka?*

  Vineyard's stream is an abstraction of a sequence of objects, where each object typically
  represents a small portion of the entire object (e.g., a mini-batch of a tensor). This
  abstraction is designed to support cross-engine pipelining between consecutive workers in
  a data analytics pipeline (e.g., a dataframe engine generating training data while the
  subsequent machine learning engine consumes the data and trains the model simultaneously).

  The primary distinction between vineyard's stream and traditional stream frameworks like
  Kafka is that data in vineyard's stream is still abstracted as (high-level) objects and
  can be consumed in a zero-copy manner, similar to normal objects in vineyard. In contrast,
  Kafka is designed for stream processing applications and abstracts data as (low-level)
  messages. Utilizing Kafka in the aforementioned scenario would still incur (de)serialization
  and memory copy costs.

7. *Does vineyard support accessing remote objects?*

  Yes, vineyard's RPC client can access the metadata of an object, regardless of whether
  the object is local or remote. This capability enables users and internal operators to
  examine essential information (e.g., chunk axis, size) about an object, assisting in
  decision-making processes related to object management (e.g., determining the need for
  repartitioning, planning the next workload).

8. *How does migration work in vineyard? Is it automatically triggered?*

  Consider a scenario where workload *A* produces a global object *O*, and the subsequent
  workload *B* consumes *O* as input. In a Kubernetes cluster with multiple hosts (e.g.,
  *h1*, *h2*, *h3*, *h4*), if *A* has two worker pods on *h1* and *h2*, the local objects
  (i.e., *O1* and *O2*) of *O* are stored on *h1* and *h2*, respectively.

  If the two worker pods of *B* (i.e., *B1* and *B2*) are placed on *h1* and *h3*, *B1*
  can access *O1* locally via memory mapping. However, *B2* (on *h3*) cannot access *O2*
  since it resides on *h2*. In this situation, a utility program distributed with vineyard
  in the :code:`initContainer` of *B2* triggers the migration of *O2* from *h2* to *h3*,
  enabling pod *B2* to access *O2* locally.

  Although data migration incurs a cost, the scheduler plugin has been developed to
  prioritize *h2* when launching *B2*, minimizing the need for migration whenever possible.

9. *What's the minimal Kubernetes version requirement for vineyard operator?*

  At present, we only test the vineyard operator based on Kubernetes 1.24.0. 
  So we highly recommend using Kubernetes 1.24.0 or above.

10. *Why the vineyard operator can't be deployed on Kubernetes?*

  If you use the helm to deploy the vineyard operator, you may find the vineyard operator
  can't be deployed successfully after a long time. In this case, you should check whether
  the command contains the flag `--wait`. If so, you should remove the flag `--wait` and
  try to install the operator again.

11. *How to connect to the vineyard cluster deployed by the vineyard operator?*

  There are two ways to connect to the vineyard cluster deployed by the vineyard operator:

  - `Through IPC`. Create a pod with the specific labels so that the pod can be scheduled 
    to the node where the vineyard cluster is deployed.
  
  - `Through RPC`. Connect to the vineyard cluster through the RPC service exposed by the 
    vineyard operator. You could refer to the `guide`_ for more details.

12. *Is there a way to install the vineyard cluster on Kubernetes quickly?*

  To reduce the complexity of the installation, we provide a `command line tool`_
  to install the vineyard cluster on Kubernetes quickly.

.. _open an issue: https://github.com/v6d-io/v6d/issues/new
.. _post it to discussions: https://github.com/v6d-io/v6d/discussions/new
.. _guide: ../../tutorials/kubernetes/using-vineyard-operator.rst
.. _command line tool: ../../notes/cloud-native/vineyardctl.md
