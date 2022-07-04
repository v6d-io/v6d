Deploying on Linux/MacOS
========================

.. _deploying-locally:

Vineyard can be deployed locally on MacOS and popular Linux distributions using
the pip-installed package with the following command:

.. code:: shell

    python -m vineyard

Command-line arguments
----------------------

Various command-line arguments are supported to configure the behaviors of the
vineyard server:

.. code:: shell

    $ python -m vineyard --help

    vineyardd: Usage: vineyardd [options]

    Flags from /tmp/vineyard-20220702-50760-10zgmhv/v6d-0.6.0/src/server/memory/dlmalloc.cc:
        -reserve_memory (Pre-reserving enough memory pages)
            type: bool
            default: false

    Flags from /tmp/vineyard-20220702-50760-10zgmhv/v6d-0.6.0/src/server/util/spec_resolvers.cc:
        -deployment (deployment mode: local, distributed)
            type: string
            default: "local"
        -etcd_cmd (path of etcd executable)
            type: string
            default: ""
        -etcd_endpoint (endpoint of etcd)
            type: string
            default: "http://127.0.0.1:2379"
        -etcd_prefix (path prefix in etcd)
            type: string
            default: "vineyard"
        -meta (Metadata storage, can be one of: etcd, local)
            type: string
            default: "etcd"
        -metrics (Alias for --prometheus, and takes precedence over --prometheus)
            type: bool
            default: false
        -prometheus (Whether to print metrics for prometheus or not)
            type: bool
            default: false
        -rpc (Enable RPC service by default)
            type: bool
            default: true
        -rpc_socket_port (port to listen in rpc server)
            type: int32
            default: 9600
        -size (shared memory size for vineyardd, the format could be 1024M, 1024000, 1G, or 1Gi)
            type: string
            default: "256Mi"
        -socket (IPC socket file location)
            type: string
            default: "/var/run/vineyard.sock"
        -stream_threshold (memory threshold of streams (percentage of total memory))
            type: int64
            default: 80
        -sync_crds (Synchronize CRDs when persisting objects)
            type: bool
            default: false
