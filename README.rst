.. raw:: html

    <h1 align="center">
        <img src="https://v6d.io/_static/vineyard_logo.png" width="397" alt="vineyard">
    </h1>
    <p align="center">
        an in-memory immutable data manager
    </p>

|Build and Test| |Coverage| |Docs| |FAQ| |Artifact HUB| |License| |CII Best Practices| |FOSSA|


Vineyard (v6d) is an in-memory immutable data manager
that provides **out-of-the-box high-level** abstraction and **zero-copy in-memory** sharing for
distributed data in big data tasks, such as graph analytics (e.g., `GraphScope`_), numerical
computing (e.g., `Mars`_), and machine learning.

.. image:: https://v6d.io/_static/cncf-small.png
  :width: 400
  :alt: Vineyard is a CNCF sandbox project

Vineyard is a `CNCF sandbox project`_ and indeed made successful by its community.

What is vineyard
----------------

Vineyard is designed to enable zero-copy data sharing between big data systems.
Let's begin with a typical machine learning task of `time series prediction with LSTM`_.
We can see that the task is divided into steps of works:

- First, we read the data from the file system as a ``pandas.DataFrame``.
- Then, we apply some preprocessing jobs, such as eliminating null values to the dataframe.
- After that, we define the model, and train the model on the processed dataframe
  in PyTorch.
- Finally, the performance of the model is evaluated.

On a single machine, although pandas and PyTorch are two different systems targeting different tasks,
data can be shared between them efficiently with little extra-cost, with everything happening
end-to-end in a single python script.

.. image:: https://v6d.io/_static/vineyard_compare.png
   :alt: Comparing the workflow with and without vineyard

What if the input data is too big to be processed on a single machine?
As illustrated on the left side of the figure, a common practice is to store the data as tables on
a distributed file system (e.g., HDFS), and replace ``pandas`` with ETL processes using SQL over a
big data system such as Hive and Spark. To share the data with PyTorch, the intermediate results are
typically saved back as tables on HDFS. This can bring some headaches to developers.

1. For the same task, users are forced to program for multiple systems (SQL & Python).

2. Data could be polymorphic. Non-relational data, such as tensors, dataframes and graphs/networks (in `GraphScope`_) are
   becoming increasingly prevalent. Tables and SQL may not be best way to store/exchange or process them.
   Having the data transformed from/to "tables" back and forth between different systems could be a huge
   overhead.

3. Saving/loading the data to/from the external storage
   requires lots of memory-copies and IO costs.

Vineyard is designed to solve these issues by providing:

1. **In-memory** distributed data sharing in a **zero-copy** fashion to avoid
   introducing extra I/O costs by exploiting a shared memory manager derived from plasma.

2. Built-in **out-of-box high-level** abstraction to share the distributed
   data with complex structures (e.g., distributed graphs)
   with nearly zero extra development cost, while the transformation costs are eliminated.

As shown in the right side of the above figure, we illustrate how to integrate
vineyard to solve the task in the big data context.

First, we use `Mars`_ (a tensor-based unified framework for large-scale data
computation which scales Numpy, Pandas and Scikit-learn) to preprocess the raw data
just like the single machine solution do, and save the preprocessed dataframe into vineyard.

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

Finally, we run the training phase distributedly across the cluster.

From the example, we see that with vineyard, the task in the big data context
can be handled with only minor modifications to the single machine solution. Compare
with the existing approaches, the
I/O and transformation overheads are also eliminated.

Features
---------

In-Memory immutable data sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vineyard is an in-memory immutable data manager, sharing immutable data across
different systems via shared memory without extra overheads. Vineyard eliminates
the overhead of serialization/deserialization and IO during exchanging immutable
data between systems.

Out-of-box high level data abstraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Computation frameworks usually have their own data abstractions for high-level concepts,
for example tensor could be `torch.tensor`, `tf.Tensor`, `mxnet.ndarray` etc., not to
mention that every `graph processing engine <https://github.com/alibaba/GraphScope>`_
has its own graph structure representations.

The variety of data abstractions makes the sharing hard. Vineyard provides out-of-box
high-level data abstractions over in-memory blobs, by describing objects using hierarchical
metadatas. Various computation systems can utilize the built-in high level data abstractions
to exchange data with other systems in computation pipeline in a concise manner.

Stream pipelining
^^^^^^^^^^^^^^^^^

A computation doesn't need to wait all precedent's result arrive before starting to work.
Vineyard provides stream as a special kind of immutable data for such pipelining scenarios.
The precedent job can write the immutable data chunk by chunk to vineyard, while maintaining
the data structure semantic, and the successor job reads shared-memory chunks from vineyard's
stream without extra copy cost, then triggers it's own work. The overlapping helps for
reducing the overall processing time and memory consumption.

Drivers
^^^^^^^

Many big data analytical tasks have lots of boilerplate routines for tasks that
unrelated to the computation itself, e.g., various IO adaptors, data partition
strategies and migration jobs. As the data structure abstraction usually differs
between systems such routines cannot be easily reused.

Vineyard provides such common manipulate routines on immutable data as drivers.
Besides sharing the high level data abstractions, vineyard extends the capability
of data structures by drivers, enabling out-of-box reusable routines for the
boilerplate part in computation jobs.

Integrate with Kubernetes
-------------------------

Vineyard helps share immutable data between different workloads, is a natural fit
to cloud-native computing. Vineyard could provide efficient distributed data sharing
in cloud-native environment by embracing cloud-native big data processing and Kubernetes
helps vineyard leverage the scale-in/out and scheduling ability of Kubernetes.

Deployment
^^^^^^^^^^

For better leveraging the scale-in/out capability of Kubernetes for worker pods of
a data analytical job, vineyard could be deployed on Kubernetes to as a DaemonSet
in Kubernetes cluster. Vineyard pods shares memory with worker pods using a UNIX
domain socket with fine-grained access control.

The UNIX domain socket can be either mounted on ``hostPath`` or via a ``PersistentVolumeClaim``.
When users bundle vineyard and the workload to the same pod, the UNIX domain socket
could also be shared using an ``emptyDir``.

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

Install vineyard
----------------

Vineyard is distributed as a `python package`_ and can be easily installed with ``pip``:

.. code:: shell

   pip3 install vineyard

The latest version of online documentation can be found at https://v6d.io.

If you want to build vineyard from source, please refer to `Installation`_.

FAQ
---

Vineyard shares many similarities with other opensource projects, but still differs
a lot with them. We are frequently asked with the following questions about vineyard,

* Q: Can clients look at the data while the stream is being filled?

  One piece of data for multiple clients is one of the target scenarios as the
  data live in vineyard is *immutable*, and multiple clients can safely consume
  the same piece of data by memory sharing, without the extra cost and extra memory
  usage of copying data back and forth.

* Q: How vineyard avoids serialization/deserialization between systems in different languages?

  Vineyard provides higher-level data abstractions (e.g., ndarrays, dataframes) that
  could be shared in a natural way between different processes.

* . . . . . .

For more detailed information, please refer to our `FAQ`_ page.

Getting involved
----------------

- Join in the `CNCF Slack`_ and navigate to the ``#vineyard`` channel for discussion.
- Read `contribution guide`_.
- Please report bugs by submitting a `GitHub issue`_ or ask me anything in `Github discussion`_.
- Submit contributions using pull requests.

Thank you in advance for your contributions to vineyard!

Acknowledgements
----------------

We thank the following excellent opensource projects:

- `apache-arrow <https://github.com/apache/arrow>`_, a cross-language development platform for in-memory analytics;
- `boost-leaf <https://github.com/boostorg/leaf>`_, a C++ lightweight error augmentation framework;
- `dlmalloc <http://gee.cs.oswego.edu/dl/html/malloc.htmlp>`_, Doug Lea's memory allocator;
- `etcd-cpp-apiv3 <https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3>`_, a C++ API for etcd's v3 client API;
- `flat_hash_map <https://github.com/skarupke/flat_hash_map>`_, an efficient hashmap implementation;
- `jemalloc <https://github.com/jemalloc/jemalloc>`_ a general purpose ``malloc(3)`` implementation.
- `nlohmann/json <https://github.com/nlohmann/json>`_, a json library for modern c++.
- `pybind11 <https://github.com/pybind/pybind11>`_, a library for seamless operability between C++11 and Python;
- `s3fs <https://github.com/dask/s3fs>`_, a library provide a convenient Python filesystem interface for S3.
- `tbb <https://github.com/oneapi-src/oneTBB>`_ a C++ library for threading building blocks.

License
-------

**Vineyard** is distributed under `Apache License 2.0`_. Please note that
third-party libraries may not have the same license as vineyard.

|FOSSA Status|

.. _Mars: https://github.com/mars-project/mars
.. _GraphScope: https://github.com/alibaba/GraphScope
.. _Installation: https://github.com/v6d-io/v6d/blob/main/docs/notes/install.rst
.. _Apache License 2.0: https://github.com/v6d-io/v6d/blob/main/LICENSE
.. _contribution guide: https://github.com/v6d-io/v6d/blob/main/CONTRIBUTING.rst
.. _time series prediction with LSTM: https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter5_RNN/time-series/lstm-time-series.ipynb
.. _python package: https://pypi.org/project/vineyard/
.. _CNCF Slack: https://slack.cncf.io/
.. _GitHub issue: https://github.com/v6d-io/v6d/issues/new
.. _Github discussion: https://github.com/v6d-io/v6d/discussions/new
.. _FAQ: https://v6d.io/notes/faq.html
.. _CNCF sandbox project: https://www.cncf.io/sandbox-projects/

.. |Build and Test| image:: https://github.com/v6d-io/v6d/workflows/Build%20and%20Test/badge.svg
   :target: https://github.com/v6d-io/v6d/actions?workflow=Build%20and%20Test
.. |Coverage| image:: https://codecov.io/gh/v6d-io/v6d/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/v6d-io/v6d
.. |Docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://v6d.io
.. |FAQ| image:: https://img.shields.io/badge/-FAQ-blue?logo=Read%20The%20Docs
   :target: https://v6d.io/notes/faq.html
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
