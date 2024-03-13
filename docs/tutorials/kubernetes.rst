Vineyard on Kubernetes
======================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   ./kubernetes/using-vineyard-operator.rst
   ./kubernetes/ml-pipeline-mars-pytorch.rst
   ./kubernetes/data-sharing-with-vineyard-on-kubernetes.rst
   ./kubernetes/efficient-data-sharing-in-kubeflow-with-vineyard-csi-driver.rst
   ./kubernetes/vineyard-on-fluid.rst

Vineyard can be seamlessly deployed on Kubernetes, managed by the :ref:`vineyard-operator`,
to enhance big-data workflows through its data-aware scheduling policy. This policy
orchestrates shared objects and routes jobs to where their input data resides. In the
following tutorials, you will learn how to deploy Vineyard and effectively integrate it
with Kubernetes.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./kubernetes/using-vineyard-operator
      :type: ref
      :text: Vineyard operator
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The Vineyard operator serves as the central component for seamless integration with Kubernetes.

   ---

   .. link-button:: ./kubernetes/ml-pipeline-mars-pytorch
      :type: ref
      :text: ML with Vineyard
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Vineyard functions as an efficient intermediate data storage solution for machine learning pipelines on Kubernetes.
