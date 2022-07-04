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

Deployment with Helm
^^^^^^^^^^^^^^^^^^^^

Vineyard also has tight integration with Kubernetes and Helm. Vineyard can be deployed
with ``helm``:

.. code:: shell

   helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
   helm install vineyard vineyard/vineyard

In the further vineyard will improve the integration with Kubernetes by abstract
vineyard objects as as Kubernetes resources (i.e., CRDs), and leverage a vineyard
operator to operate vineyard cluster.
