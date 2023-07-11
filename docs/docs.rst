.. vineyard documentation master file, created by
   sphinx-quickstart on Tue Aug 27 10:19:05 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: Vineyard (v6d), a CNCF sandbox project, is an in-memory immutable data manager
      that provides **out-of-the-box high-level** abstraction and **zero-copy in-memory** sharing for
      distributed data in big data tasks, such as graph analytics (e.g., `GraphScope`_), numerical
      computing (e.g., `Mars`_), and machine learning.
   :keywords: distributed-systems, distributed, shared-memory, graph-analytics, in-memory-storage,
              big-data-analytics, distributed-comp

.. figure:: images/vineyard-logo-rect.png
   :width: 397
   :alt: Vineyard: an in-memory immutable data manager
   :target: https://v6d.io

   *an in-memory immutable data manager*

|PyPI| |FAQ| |Discussion| |Slack| |License| |ACM DL|

Why bother?
-----------

Sharing intermediate data between systems in modern big data and AI workflows
can be challenging, often causing significant bottlenecks in such jobs. Let's
consider the following fraud detection pipeline:

.. figure:: images/fraud-detection-job.jpg
   :width: 75%
   :alt: A real-life fraud detection job

   A real-life fraud detection job

From the pipeline, we observed:

1. Users usually prefer to program with dedicated computing systems for different tasks in the
   same applications, such as SQL and Python.

   **Integrating a new computing system into production environments demands high technical
   effort to align with existing production environments in terms of I/O, failover, etc.**

2. Data could be polymorphic. Non-relational data, such as tensors, dataframes (in Pandas) and
   graphs/networks (in `GraphScope`_) are becoming increasingly prevalent. Tables and SQL may
   not be the best way to store, exchange, or process them.

   **Transforming the data back and forth between different systems as "tables" could
   result in a significant overhead.**

3. Saving/loading the data to/from the external storage requires numerous memory copies and
   incurs high IO costs.

What is Vineyard?
-----------------

Vineyard (v6d) is an **in-memory immutable data manager** that offers **out-of-the-box high-level**
abstraction and **zero-copy sharing** for distributed data in big data tasks, such as
graph analytics (e.g., `GraphScope`_), numerical computing (e.g., `Mars`_), and machine learning.

Features
^^^^^^^^

Efficient data sharing
~~~~~~~~~~~~~~~~~~~~~~

Vineyard shares immutable data across different systems using shared memory without extra overheads,
eliminating the overhead of serialization/deserialization and IO when exchanging immutable
data between systems.

Out-of-the-box data abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vineyard defines a metadata-payload separated data model to capture the payload commonalities and
method commonalities between sharable objects in different programming languages and different
computing systems in a unified way.

The :ref:`VCDL` (Vineyard Component Description Language) is specifically designed to annotate
sharable members and methods, enabling automatic generation of boilerplate code for minimal
integration effort.

Pluggable I/O routines
~~~~~~~~~~~~~~~~~~~~~~

In many big data analytical tasks, a substantial portion of the workload consists of boilerplate
routines that are unrelated to the core computation. These routines include various IO adapters,
data partition strategies, and migration jobs. Due to different data structure abstractions across
systems, these routines are often not easily reusable, leading to increased complexity and redundancy.

Vineyard provides common manipulation routines for immutable data as drivers, which extend
the capabilities of data structures by registering appropriate drivers. This enables out-of-the-box
reuse of boilerplate components across diverse computation jobs.

Data orchestration on Kubernetes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vineyard provides efficient distributed data sharing in cloud-native environments by embracing
cloud-native big data processing. Kubernetes helps Vineyard leverage the scale-in/out and
scheduling abilities of Kubernetes.

Use cases
^^^^^^^^^

.. panels::
   :header: text-center
   :container: container-lg pb-4
   :column: col-lg-4 col-md-4 col-sm-4 col-xs-12 p-2
   :body: text-center

   .. link-button:: #
      :type: url
      :text: Object manager
      :classes: btn-block stretched-link

   Put and get arbitrary objects using Vineyard, in a zero-copy way!

   ---

   .. link-button:: #
      :type: url
      :text: Cross-system sharing
      :classes: btn-block stretched-link

   Share large objects across computing systems.

   ---

   .. link-button:: #
      :type: url
      :text: Data orchestration
      :classes: btn-block stretched-link

   Vineyard coordinates the flow of objects and jobs on Kubernetes based on data-aware scheduling.

Get started now!
----------------

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: notes/getting-started
      :type: ref
      :text: User Guides
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Get started with Vineyard.

   ---

   .. link-button:: notes/cloud-native/deploy-kubernetes
      :type: ref
      :text: Deploy on Kubernetes
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Deploy Vineyard on Kubernetes and accelerate big-data analytical workflows on cloud-native
   infrastructures.

   ---

   .. link-button:: tutorials/tutorials
      :type: ref
      :text: Tutorials
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Explore use cases and tutorials where Vineyard can bring added value.

   ---

   .. link-button:: notes/developers
      :type: ref
      :text: Getting Involved
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Get involved and become part of the Vineyard community.

   ---

   .. link-button:: notes/developers/faq
      :type: ref
      :text: FAQ
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Frequently asked questions and discussions during the adoption of Vineyard.

Read the Paper
--------------

- Wenyuan Yu, Tao He, Lei Wang, Ke Meng, Ye Cao, Diwen Zhu, Sanhong Li, Jingren Zhou.
  `Vineyard: Optimizing Data Sharing in Data-Intensive Analytics <https://v6d.io/vineyard-sigmod-2023.pdf>`_.
  ACM SIG Conference on Management of Data (SIGMOD), industry, 2023. |ACM DL|.

Vineyard is a `CNCF sandbox project`_ and is made successful by its community.

.. image:: https://v6d.io/_static/cncf-color.svg
   :width: 400
   :alt: Vineyard is a CNCF sandbox project

.. toctree::
   :maxdepth: 1
   :caption: User Guides
   :hidden:

   notes/getting-started.rst
   notes/architecture.rst
   notes/key-concepts.rst

.. toctree::
   :maxdepth: 1
   :caption: Cloud-Native
   :hidden:

   notes/cloud-native/deploy-kubernetes.rst
   notes/cloud-native/vineyard-operator.rst
   Command-line tool <notes/cloud-native/vineyardctl.md>

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/data-processing.rst
   tutorials/kubernetes.rst
   tutorials/extending.rst

.. toctree::
   :maxdepth: 1
   :caption: Integration
   :hidden:

   notes/integration-bigdata.rst
   notes/integration-orchestration.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   notes/references.rst

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides
   :hidden:

   notes/developers.rst
   notes/developers/faq.rst

.. _Mars: https://github.com/mars-project/mars
.. _GraphScope: https://github.com/alibaba/GraphScope
.. _CNCF sandbox project: https://www.cncf.io/sandbox-projects/

.. |PyPI| image:: https://img.shields.io/pypi/v/vineyard?color=blue
   :target: https://pypi.org/project/vineyard
.. |FAQ| image:: https://img.shields.io/badge/-FAQ-blue?logo=Read%20The%20Docs
   :target: https://v6d.io/notes/faq.html
.. |Discussion| image:: https://img.shields.io/badge/Discuss-Ask%20Questions-blue?logo=GitHub
   :target: https://github.com/v6d-io/v6d/discussions
.. |Slack| image:: https://img.shields.io/badge/Slack-Join%20%23vineyard-purple?logo=Slack
   :target: https://slack.cncf.io/
.. |License| image:: https://img.shields.io/github/license/v6d-io/v6d
   :target: https://github.com/v6d-io/v6d/blob/main/LICENSE

.. |ACM DL| image:: https://img.shields.io/badge/ACM%20DL-10.1145%2F3589780-blue
   :target: https://dl.acm.org/doi/10.1145/3589780
