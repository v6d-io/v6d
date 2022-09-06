Tutorials
=========

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   tutorials/distributed-learning.rst
   tutorials/adding-custom-datatypes-cpp.rst

We showcase step-by-step case studies of how to combine the functionalities of vineyard
with existing data-intensive jobs. We show that vineyard can bring huge gains in both
performance and conveniences when users have a complex workflow that involves multiple
computing engines.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: tutorials/distributed-learning
      :type: ref
      :text: Distributed Learning
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   How vineyard can help in a distributed machine learning training workflow where
   various computing engine are involved.

Besides, vineyard has implemented a set of efficient data structures that needed in common
data-intensive jobs, e.g., tensors, data frames, tables and graphs. The data types can be
extended as well in a fairly straightforward way. Once registered the user defined custom
types into the vineyard type registry, computing engines run on top of vineyard can immediately
gain the benefits brought by vineyard.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: tutorials/adding-custom-datatypes-cpp
      :type: ref
      :text: Adding Custom Types
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Adding new data types and register to vineyard's builder/resolver context.
