.. raw:: html

    <h1 align="center" style="clear: both;">
        <img src="https://v6d.io/_static/vineyard-logo-rect.png" width="397" alt="vineyard">
    </h1>
    <p align="center">
        an in-memory immutable data manager
    </p>

|Vineyard CI| |Coverage| |Docs| |FAQ| |Discussion| |Slack| |License| |CII Best Practices| |FOSSA|

|PyPI| |crates.io| |Docker HUB| |Artifact HUB| |ACM DL|

Vineyard (v6d) is an innovative in-memory immutable data manager that offers **out-of-the-box
high-level** abstractions and **zero-copy in-memory** sharing for distributed data in various
big data tasks, such as graph analytics (e.g., `GraphScope`_), numerical computing
(e.g., `Mars`_), and machine learning.

.. image:: https://v6d.io/_static/cncf-color.svg
  :width: 400
  :alt: Vineyard is a CNCF sandbox project

Vineyard is a `CNCF sandbox project`_ and indeed made successful by its community.

Table of Contents
-----------------

* `Overview <#what-is-vineyard>`_
* `Features of vineyard <#features>`_

  * `Efficient sharing for in-memory immutable data <#in-memory-immutable-data-sharing>`_
  * `Out-of-the-box high level data structures <#out-of-the-box-high-level-data-abstraction>`_
  * `Pipelining using stream <#stream-pipelining>`_
  * `I/O Drivers <#drivers>`_

* `Getting started with Vineyard <#try-vineyard>`_
* `Deploying on Kubernetes <#deploying-on-kubernetes>`_
* `Frequently asked questions <#faq>`_
* `Getting involved in our community <#getting-involved>`_
* `Third-party dependencies <#acknowledgements>`_

What is vineyard
----------------

Vineyard is specifically designed to facilitate zero-copy data sharing among big data systems. To
illustrate this, let's consider a typical machine learning task of `time series prediction with LSTM`_.
This task can be broken down into several steps:

- First, we read the data from the file system as a ``pandas.DataFrame``.
- Next, we apply various preprocessing tasks, such as eliminating null values, to the dataframe.
- Once the data is preprocessed, we define the model and train it on the processed dataframe using PyTorch.
- Finally, we evaluate the performance of the model.

In a single-machine environment, pandas and PyTorch, despite being two distinct systems designed for
different tasks, can efficiently share data with minimal overhead. This is achieved through an
end-to-end process within a single Python script.

.. image:: https://v6d.io/_static/vineyard_compare.png
   :alt: Comparing the workflow with and without vineyard

What if the input data is too large to be processed on a single machine?

   As depicted on the left side of the figure, a common approach is to store the data as tables in
   a distributed file system (e.g., HDFS) and replace ``pandas`` with ETL processes using SQL over a
   big data system such as Hive and Spark. To share the data with PyTorch, the intermediate results are
   typically saved back as tables on HDFS. However, this can introduce challenges for developers.

1. For the same task, users must program for multiple systems (SQL & Python).

2. Data can be polymorphic. Non-relational data, such as tensors, dataframes, and graphs/networks
   (in `GraphScope`_) are becoming increasingly common. Tables and SQL may not be the most efficient
   way to store, exchange, or process them. Transforming the data from/to "tables" between different
   systems can result in significant overhead.

3. Saving/loading the data to/from external storage
   incurs substantial memory-copies and IO costs.

Vineyard addresses these issues by providing:

1. **In-memory** distributed data sharing in a **zero-copy** fashion to avoid
   introducing additional I/O costs by leveraging a shared memory manager derived from plasma.

2. Built-in **out-of-the-box high-level** abstractions to share distributed
   data with complex structures (e.g., distributed graphs)
   with minimal extra development cost, while eliminating transformation costs.

As depicted on the right side of the above figure, we demonstrate how to integrate
vineyard to address the task in a big data context.

First, we utilize `Mars`_ (a tensor-based unified framework for large-scale data
computation that scales Numpy, Pandas, and Scikit-learn) to preprocess the raw data,
similar to the single-machine solution, and store the preprocessed dataframe in vineyard.

+-------------+-----------------------------------------------------------------------------+
|             | .. code-block:: python                                                      |
| single      |                                                                             |
|             |     data_csv = pd.read_csv('./data.csv', usecols=[1])                       |
+-------------+-----------------------------------------------------------------------------+
|             | .. code-block:: python                                                      |
|             |                                                                             |
|             |     import mars.dataframe as md                                             |
| distributed |     dataset = md.read_csv('hdfs://server/data_full', usecols=[1])           |
|             |     # after preprocessing, save the dataset to vineyard                     |
|             |     vineyard_distributed_tensor_id = dataset.to_vineyard()                  |
+-------------+-----------------------------------------------------------------------------+

Then, we modify the
training phase to get the preprocessed data from vineyard. Here vineyard makes
the sharing of distributed data between `Mars`_ and PyTorch just like a local
variable in the single machine solution.

+-------------+-----------------------------------------------------------------------------+
|             | .. code-block:: python                                                      |
| single      |                                                                             |
|             |     data_X, data_Y = create_dataset(dataset)                                |
+-------------+-----------------------------------------------------------------------------+
|             | .. code-block:: python                                                      |
|             |                                                                             |
|             |     client = vineyard.connect(vineyard_ipc_socket)                          |
| distributed |     dataset = client.get(vineyard_distributed_tensor_id).local_partition()  |
|             |     data_X, data_Y = create_dataset(dataset)                                |
+-------------+-----------------------------------------------------------------------------+

Finally, we execute the training phase in a distributed manner across the cluster.

From this example, it is evident that with vineyard, the task in the big data context can
be addressed with only minor adjustments to the single-machine solution. Compared to
existing approaches, vineyard effectively eliminates I/O and transformation overheads.

Features
--------

Efficient In-Memory Immutable Data Sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vineyard serves as an in-memory immutable data manager, enabling efficient data
sharing across different systems via shared memory without additional overheads.
By eliminating serialization/deserialization and IO costs during data exchange
between systems, Vineyard significantly improves performance.

Out-of-the-Box High-Level Data Abstractions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computation frameworks often have their own data abstractions for high-level concepts.
For example, tensors can be represented as `torch.tensor`, `tf.Tensor`, `mxnet.ndarray`, etc.
Moreover, every `graph processing engine <https://github.com/alibaba/GraphScope>`_
has its unique graph structure representation.

The diversity of data abstractions complicates data sharing. Vineyard addresses this
issue by providing out-of-the-box high-level data abstractions over in-memory blobs,
using hierarchical metadata to describe objects. Various computation systems can
leverage these built-in high-level data abstractions to exchange data with other systems
in a computation pipeline concisely and efficiently.

Stream Pipelining for Enhanced Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A computation doesn't need to wait for all preceding results to arrive before starting
its work. Vineyard provides a stream as a special kind of immutable data for pipelining
scenarios. The preceding job can write immutable data chunk by chunk to Vineyard while
maintaining data structure semantics. The successor job reads shared-memory chunks from
Vineyard's stream without extra copy costs and triggers its work. This overlapping
reduces the overall processing time and memory consumption.

Versatile Drivers for Common Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many big data analytical tasks involve numerous boilerplate routines that are unrelated
to the computation itself, such as various IO adapters, data partition strategies, and
migration jobs. Since data structure abstractions usually differ between systems, these
routines cannot be easily reused.

Vineyard provides common manipulation routines for immutable data as drivers.
In addition to sharing high-level data abstractions, Vineyard extends the capability
of data structures with drivers, enabling out-of-the-box reusable routines for the
boilerplate parts in computation jobs.

Try Vineyard
------------

Vineyard is available as a `python package`_ and can be effortlessly installed using ``pip``:

.. code:: shell

   pip3 install vineyard

For comprehensive and up-to-date documentation, please visit https://v6d.io.

If you wish to build vineyard from source, please consult the `Installation`_ guide. For
instructions on building and running unittests locally, refer to the `Contributing`_ section.

After installation, you can initiate a vineyard instance using the following command:

.. code:: shell

   python3 -m vineyard

For further details on connecting to a locally deployed vineyard instance, please
explore the `Getting Started`_ guide.

Deploying on Kubernetes
-----------------------

Vineyard is designed to efficiently share immutable data between different workloads,
making it a natural fit for cloud-native computing. By embracing cloud-native big data
processing and Kubernetes, Vineyard enables efficient distributed data sharing in
cloud-native environments while leveraging the scaling and scheduling capabilities
of Kubernetes.

To effectively manage all components of Vineyard within a Kubernetes cluster, we have
developed the Vineyard Operator. For more information, please refer to the `Vineyard
Operator`_ documentation.

FAQ
---

Vineyard shares many similarities with other open-source projects, yet it also has
distinct features. We often receive the following questions about Vineyard:

* Q: Can clients access the data while the stream is being filled?

  Sharing one piece of data among multiple clients is a target scenario for Vineyard,
  as the data stored in Vineyard is *immutable*. Multiple clients can safely consume
  the same piece of data through memory sharing, without incurring extra costs or
  additional memory usage from copying data back and forth.

* Q: How does Vineyard avoid serialization/deserialization between systems in different
  languages?

  Vineyard provides high-level data abstractions (e.g., ndarrays, dataframes) that can
  be naturally shared between different processes, eliminating the need for serialization
  and deserialization between systems in different languages.

* . . . . . .

For more detailed information, please refer to our `FAQ`_ page.

Get Involved
------------

- Join the `CNCF Slack`_ and participate in the ``#vineyard`` channel for discussions
  and collaboration.
- Familiarize yourself with our `contribution guide`_ to understand the process of
  contributing to vineyard.
- If you encounter any bugs or issues, please report them by submitting a `GitHub
  issue`_ or engage in a conversation on `Github discussion`_.
- We welcome and appreciate your contributions! Submit them using pull requests.

Thank you in advance for your valuable contributions to vineyard!

Publications
------------

- Wenyuan Yu, Tao He, Lei Wang, Ke Meng, Ye Cao, Diwen Zhu, Sanhong Li, Jingren Zhou.
  `Vineyard: Optimizing Data Sharing in Data-Intensive Analytics <https://v6d.io/vineyard-sigmod-2023.pdf>`_.
  ACM SIG Conference on Management of Data (SIGMOD), industry, 2023. |ACM DL|.

If you use this software, please cite our paper using the following metadata:

.. code:: bibtex

   @article{yu2023vineyard,
      author = {Yu, Wenyuan and He, Tao and Wang, Lei and Meng, Ke and Cao, Ye and Zhu, Diwen and Li, Sanhong and Zhou, Jingren},
      title = {Vineyard: Optimizing Data Sharing in Data-Intensive Analytics},
      year = {2023},
      issue_date = {June 2023},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      volume = {1},
      number = {2},
      url = {https://doi.org/10.1145/3589780},
      doi = {10.1145/3589780},
      journal = {Proc. ACM Manag. Data},
      month = {jun},
      articleno = {200},
      numpages = {27},
      keywords = {data sharing, in-memory object store}
   }

Acknowledgements
----------------

We thank the following excellent open-source projects:

- `apache-arrow <https://github.com/apache/arrow>`_, a cross-language development platform for in-memory analytics.
- `boost-leaf <https://github.com/boostorg/leaf>`_, a C++ lightweight error augmentation framework.
- `cityhash <https://github.com/google/cityhash>`_, CityHash, a family of hash functions for strings.
- `dlmalloc <http://gee.cs.oswego.edu/dl/html/malloc.htmlp>`_, Doug Lea's memory allocator.
- `etcd-cpp-apiv3 <https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3>`_, a C++ API for etcd's v3 client API.
- `flat_hash_map <https://github.com/skarupke/flat_hash_map>`_, an efficient hashmap implementation.
- `gulrak/filesystem <https://github.com/gulrak/filesystem>`_, an implementation of C++17 std::filesystem.
- `libcuckoo <https://github.com/efficient/libcuckoo>`_, libcuckoo, a high-performance, concurrent hash table.
- `mimalloc <https://github.com/microsoft/mimalloc>`_, a general purpose allocator with excellent performance characteristics.
- `nlohmann/json <https://github.com/nlohmann/json>`_, a json library for modern c++.
- `pybind11 <https://github.com/pybind/pybind11>`_, a library for seamless operability between C++11 and Python.
- `s3fs <https://github.com/dask/s3fs>`_, a library provide a convenient Python filesystem interface for S3.
- `skywalking-infra-e2e <https://github.com/apache/skywalking-infra-e2e>`_ A generation End-to-End Testing framework.
- `skywalking-swck <https://github.com/apache/skywalking-swck>`_ A kubernetes operator for the Apache Skywalking.
- `wyhash <https://github.com/alainesp/wy>`_, C++ wrapper around wyhash and wyrand.
- `BBHash <https://github.com/rizkg/BBHash>`_, a fast, minimal-memory perfect hash function.
- `rax <https://github.com/antirez/rax>`_, an ANSI C radix tree implementation.
- `MurmurHash3 <https://github.com/aappleby/smhasher>`_, a fast non-cryptographic hash function.

License
-------

**Vineyard** is distributed under `Apache License 2.0`_. Please note that
third-party libraries may not have the same license as vineyard.

|FOSSA Status|

.. _Mars: https://github.com/mars-project/mars
.. _GraphScope: https://github.com/alibaba/GraphScope
.. _Installation: https://github.com/v6d-io/v6d/blob/main/docs/notes/developers/build-from-source.rst
.. _Contributing: https://github.com/v6d-io/v6d/blob/main/CONTRIBUTING.rst
.. _Getting Started: https://v6d.io/notes/getting-started.html
.. _Vineyard Operator: https://v6d.io/notes/cloud-native/vineyard-operator.html
.. _Apache License 2.0: https://github.com/v6d-io/v6d/blob/main/LICENSE
.. _contribution guide: https://github.com/v6d-io/v6d/blob/main/CONTRIBUTING.rst
.. _time series prediction with LSTM: https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/time-series/lstm-time-series.ipynb
.. _python package: https://pypi.org/project/vineyard/
.. _CNCF Slack: https://slack.cncf.io/
.. _GitHub issue: https://github.com/v6d-io/v6d/issues/new
.. _Github discussion: https://github.com/v6d-io/v6d/discussions/new
.. _FAQ: https://v6d.io/notes/faq.html
.. _CNCF sandbox project: https://www.cncf.io/sandbox-projects/

.. |Vineyard CI| image:: https://github.com/v6d-io/v6d/actions/workflows/build-test.yml/badge.svg
   :target: https://github.com/v6d-io/v6d/actions/workflows/build-test.yml
.. |Coverage| image:: https://codecov.io/gh/v6d-io/v6d/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/v6d-io/v6d
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://v6d.io
.. |FAQ| image:: https://img.shields.io/badge/-FAQ-blue?logo=Read%20The%20Docs
   :target: https://v6d.io/notes/faq.html
.. |Discussion| image:: https://img.shields.io/badge/Discuss-Ask%20Questions-blue?logo=GitHub
   :target: https://github.com/v6d-io/v6d/discussions
.. |Slack| image:: https://img.shields.io/badge/Slack-Join%20%23vineyard-purple?logo=Slack
   :target: https://slack.cncf.io/
.. |PyPI| image:: https://img.shields.io/pypi/v/vineyard?color=blue
   :target: https://pypi.org/project/vineyard
.. |crates.io| image:: https://img.shields.io/crates/v/vineyard.svg
   :target: https://crates.io/crates/vineyard
.. |Docker HUB| image:: https://img.shields.io/badge/docker-ready-blue.svg
   :target: https://hub.docker.com/u/vineyardcloudnative
.. |Artifact HUB| image:: https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard
   :target: https://artifacthub.io/packages/helm/vineyard/vineyard
.. |CII Best Practices| image:: https://bestpractices.coreinfrastructure.org/projects/4902/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/4902
.. |FOSSA| image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fv6d-io%2Fv6d.svg?type=shield
   :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fv6d-io%2Fv6d?ref=badge_shield
.. |FOSSA Status| image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fv6d-io%2Fv6d.svg?type=large
   :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fv6d-io%2Fv6d?ref=badge_large
.. |License| image:: https://img.shields.io/github/license/v6d-io/v6d
   :target: https://github.com/v6d-io/v6d/blob/main/LICENSE

.. |ACM DL| image:: https://img.shields.io/badge/ACM%20DL-10.1145%2F3589780-blue
   :target: https://dl.acm.org/doi/10.1145/3589780
