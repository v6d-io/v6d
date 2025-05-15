Troubleshooting
===============

This page provides guidance for addressing common issues that may arise when
working with Vineyard.

.. Installation Errors
.. -------------------

Vineyard Fails to Start
-----------------------

1. Improper Etcd Configuration

    If you encounter the following error when sending requests to Vineyard:

    .. code::

        Etcd error: etcdserver: too many operations in txn request, error code: 3

    This indicates that your Etcd configuration is not set up correctly and does not support
    more than 128 operations within a single transaction. To resolve this issue, check your Etcd
    startup parameters and increase the :code:`--max-txn-ops` value, for example, to :code:`102400`.

2. bind: Permission Denied Error When Launching vineyardd

    The Vineyard server uses a UNIX-domain socket for IPC connections and memory sharing with clients.
    By default, the UNIX-domain socket is located at :code:`/var/run/vineyard.sock`, which typically
    requires root permission.

    To launch vineyardd, you can either:

    + Run the :code:`vineyardd` command with :code:`sudo`,
    + Or, specify a different location for the UNIX-domain socket that does not require root permission
      using the :code:`--socket` command line argument, e.g.,
      .. code:: bash

          python3 -m vineyard --socket=/tmp/vineyard.sock

Vineyard Issues on Kubernetes
-----------------------------

1. Etcd Pod Resource Limitations in Kubernetes Deployment

    We have observed that etcd performance may degrade when a Vineyard client persists a large
    object, particularly in Kubernetes deployments where the CPU cores of the etcd pod are limited by
    cgroups. In such cases, users should increase the CPU resources allocated to the etcd pod. For
    more information on etcd tuning, please refer to the `Hardware recommendations
    <https://etcd.io/docs/v3.4.0/op-guide/hardware/>`_ section in the etcd documentation.
