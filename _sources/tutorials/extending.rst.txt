
Extending vineyard
==================

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   ./extending/define-datatypes-python.rst
   ./extending/define-datatypes-cpp.rst

Vineyard offers a collection of efficient data structures tailored for data-intensive tasks,
such as tensors, data frames, tables, and graphs. These data types can be easily extended
to accommodate custom requirements. By registering user-defined types in the vineyard type
registry, computing engines built on top of vineyard can instantly leverage the advantages
provided by these custom data structures.

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: ./extending/define-datatypes-python
      :type: ref
      :text: Define Python Types
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Craft builders and resolvers for custom Python data types.

   ---

   .. link-button:: ./extending/define-datatypes-cpp
      :type: ref
      :text: Define C++ Types
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Implement and register custom data types in C++ for seamless integration with vineyard's ecosystem.
