
Extending vineyard
==================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   ./extending/adding-custom-datatypes-cpp.rst

Vineyard has implemented a set of efficient data structures that needed in common
data-intensive jobs, e.g., tensors, data frames, tables and graphs. The data types can be
extended as well in a fairly straightforward way. Once registered the user defined custom
types into the vineyard type registry, computing engines run on top of vineyard can immediately
gain the benefits brought by vineyard.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./extending/adding-custom-datatypes-cpp
      :type: ref
      :text: Adding Custom Types
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Adding new data types and register to vineyard's builder/resolver context.
