Roadmap
=======

Vineyard aims to serve as an open-source in-memory immutable data manager. We
cut a major release once a year, a minor release for about every two months,
and a patch release every one or two weeks.

The roadmap for major vineyard releases are listed as follows:

v0.8.0
------

Vineyard *v0.8.0* will deliver the first implementation of the following
important features and will be hopefully release in the later Aug, 2022:

- Filesystem view of vineyard objects: vineyard objects can be accessed like
  files on a filesystem in a high-performance fashion. Such a feature would
  greatly ease the integration of computing processes with vineyard.
- Copy-on-realloc and data lineage: the mutation support would be extended
  from blobs to general objects with a carefully concurrency-control design.
- Transparent object spilling: objects in vineyard can be spilled to disk
  when they are too large to fit in memory.
- Sharing GPU memory between processes of different compute engines: we are
  working on shared memory on devices to enable boarder applications that
  can benefit from the shared vineyard store, especially for deep learning
  frameworks and GNN frameworks.

v0.7.0
------

Vineyard *v0.7.0* will be released  in later July, 2022. Vineyard v0.7.0 will
introduces the following experimental features to ease the integration of
various kinds of workloads with Vineyard:

- Limited mutation support on blobs: starts from vineyard *v0.7.0*, unsealed
  blobs can be get by other clients with an :code:`unsafe` flag to ease the
  integration of some online storage engines.
- Limited support for remote data accessing using the RPC client: vineyard
  *v0.7.0* will bring the feature about creating and accessing remote blobs
  using the RPC client. It would be greatly helpful for some specific deployment
  and the cost of remote data sourcing to vineyard is tolerable.

v0.6.0
------

We plan to release the *v0.6.0* version before tne end of June, 2022. The *v0.6.0*
release will include the following enhancement:

- Better compatibility on various platforms (e.g., CentOS and ArchLinux), and process
  platform-specific features like `LD_LIBRARY_PATH` and `libunwind` dependency
  carefully.
- Ensure the backwards compatibility with various third-party integrations, e.g.,
  apache-airflow.
- Vineyard v0.6.0 will be available from `homebrew <https://brew.sh/>`_.

v0.5.0
------

We plan to release the first preliminary version for the Rust SDK and Go SDK
in vineyard *v0.5.0*, that is expected to be delivered in later May, 2022.

In vineyard *v0.5.0*, we will investigate the opportunity about code generation
based on the metadata of vineyard objects, i.e., we could generate the data
structure definition based on the structure of metadata in runtime (for Python)
and in compile time (even maybe in runtime) for C++ and Rust.

The integration with Kubernetes (especially the CSI part) will be another key
improvement for *v0.5.0*.

Further details about release for *v0.5.0* will be added later.

v0.4.0
------

The release of vineyard *v0.4.0*, will be hopefully released before April, 2022, will
be a follow-up bugfix releases after *v0.3.0*. The version *v0.4.0* makes the
kubernetes related components better.

+ Improve the robustness of the scheduler plugin.
+ Refine the definition of CRDs.
+ Distribute the vineyard operator to artifact hub as a chart, to make it available for more users.

v0.3.0
------

We plan to release *v0.3.0* by the end of 2021. vineyard *v0.3.0* will be the first major
stable releases with fully kubernetes support, which will include:

+ A stable CRD definition for ``LocalObject`` and ``GlobalObject`` to represents vineyard objects
  as kubernetes resources.
+ A full-features scheduler plugin for kubernetes, as well as a custom controller that manages
  objects (custom resources) in vineyard cluster.
+ A refined version of Helm integration.
+ Application-aware far memory will be included in v0.3.0 as an experimental feature.

v0.2.0
------

Vineyard *v0.2.0* will address the issue about Python ecosystem compatibility, I/O, and
the kubernetes integration. Vineyard v0.2.0 will take about half of a year with several bugfix
release to testing the design and APIs to reach a stable stable state.

+ Vineyard *v0.2.0* will support any *filesystem-spec*-compatible data source/sink as well as file
  format.
+ Vineyard *v0.2.0* will support Python ecosystem (especially numpy and pandas) better.
+ Vineyard *v0.2.0* will include basic Helm integration for deploying on Kubernetes as a ``DaemonSet``.
+ A prototype of scheduler plugin to do data locality scheduling will be included into vineyard v0.2.0
  to demonstrates the capability about co-scheduling job and data in kubernetes brought by vineyard.
+ Match the criterion of CNCF sandbox project.

v0.1.0
------

Vineyard *v0.1.0* is the first release after open source. This version includes:

+ Complete functionality for both server and client.
+ Complete Python SDK.
+ User-friendly package distribution on pypi (for python SDK) and on dockerhub (for vineyardd server).

Release Notes
-------------

For more details about what changes happened for every version, please refer to
our `releases notes`_ as well.

.. _releases notes: https://github.com/v6d-io/v6d/releases
