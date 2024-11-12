Key Concepts
============

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   key-concepts/objects.rst
   key-concepts/vcdl.rst
   key-concepts/data-accessing.rst
   key-concepts/streams.rst
   key-concepts/io-drivers.rst

The *User Guide* section offers an in-depth understanding of vineyard's design
and implementation. It covers detailed environment setup instructions, the
architecture, and the core features within the vineyard engine.

*Additional information on vineyard's internals will be provided soon*.

.. tip::

   If you are new to vineyard, we recommend starting with the
   `Getting Started <https://v6d.io/notes/getting-started.html>`_ page for a smoother
   introduction.

Concepts
--------

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: key-concepts/objects
      :type: ref
      :text: Vineyard Objects
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The design space of vineyard objects.

   ---

   .. link-button:: key-concepts/vcdl
      :type: ref
      :text: VCDL and Integration
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   VCDL and how to integration vineyard with computing systems.

   ---

   .. link-button:: key-concepts/data-accessing
      :type: ref
      :text: Accessing Objects in Vineyard
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The approaches that can be used to access various kinds of objects stored in
   vineyard.

   ---

   .. link-button:: key-concepts/streams
      :type: ref
      :text: Stream in Vineyard
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The stream abstraction upon the immutable data sharing storage and its usages.

   ---

   .. link-button:: key-concepts/io-drivers
      :type: ref
      :text: I/O Drivers
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Design and implementation of the builtin I/O drivers that eases the integration
   of computing engines to existing infrastructure.
