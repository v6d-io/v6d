.. _faqs:

FAQs
====

This document collects frequently asked questions (FAQs) about the Vineyard Operator.

What's the minimal Kubernetes version requirement?
---------------------------------------------------

At present, we only test the vineyard operator based on Kubernetes 1.24.0. 
So we highly recommend using Kubernetes 1.24.0 or above.

Why the vineyard operator can't be deployed on Kubernetes?
---------------------------------------------------------------

If you deploy the vineyard operator on Kubernetes, you may encounter the following error:

.. code:: shell

    resource mapping not found for name: "vineyard-serving-cert" namespace: "vineyard-system" from "STDIN": no matches for kind "Certificate" in version "cert-manager.io/v1"
    ensure CRDs are installed first
    resource mapping not found for name: "vineyard-selfsigned-issuer" namespace: "vineyard-system" from "STDIN": no matches for kind "Issuer" in version "cert-manager.io/v1"
    ensure CRDs are installed first

It is because the vineyard operator depends on the `cert-manager`_ to 
generate the TLS certificate for the vineyard cluster. So you need to install the cert-manager first.

How to connect to the vineyard cluster deployed by the vineyard operator?
--------------------------------------------------------------------------

There are two ways to connect to the vineyard cluster deployed by the vineyard operator:
- `Through IPC`. Create a pod with the specific labels so that the pod can be scheduled to the node where 
    the vineyard cluster is deployed.
- `Through RPC`. Connect to the vineyard cluster through the RPC service exposed by the vineyard operator.
You could refer to the `guide`_ for more details.

Is there a way to install the vineyard cluster on Kubernetes quickly?
----------------------------------------------------------------------

To reduce the complexity of the installation, we provide a `command line tool`_
to install the vineyard cluster on Kubernetes quickly.

.. _cert-manager: https://cert-manager.io/
.. _guide: ../../tutorials/kubernetes/using-vineyard-operator.rst
.. _command line tool: ../../../k8s/cmd/README.md