User Guide
==========

.. toctree::
   :maxdepth: 1
   :caption: TOC
   :hidden:

   notes/install.rst
   notes/architecture.rst
   notes/objects.rst
   notes/data-accessing.rst
   notes/streams.rst
   notes/io-drivers.rst

The *User Guide* sections provides a comprehensive perspective of the design and
implementation of vineyard. Including a detailed environment setup guidance, the
architecture, as well as the core features inside the vineyard engine.

*More details about the internals of vineyard will be added soon*.

.. tip::

   If this is the first time using vineyard, checking out the
   `Getting Started <https://v6d.io/notes/getting-started.html>`_ page first would
   be better.

Installation
------------

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: notes/install
      :type: ref
      :text: Installation
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   How vineyard can be installed on various platforms.

Concepts
--------

.. panels::
   :header: text-center
   :column: col-lg-12 p-2

   .. link-button:: notes/architecture
      :type: ref
      :text: Architecture
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Overview the motivation and architecture of vineyard.

   ---

   .. link-button:: notes/objects
      :type: ref
      :text: Vineyard Objects
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The design space of vineyard objects.

   ---

   .. link-button:: notes/data-accessing
      :type: ref
      :text: Accessing Objects in Vineyard
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The approaches that can be used to access various kinds of objects stored in
   vineyard.

   ---

   .. link-button:: notes/streams
      :type: ref
      :text: Stream in Vineyard
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   The stream abstraction upon the immutable data sharing storage and its usages.

   ---

   .. link-button:: notes/io-drivers
      :type: ref
      :text: I/O Drivers
      :classes: btn-block stretched-link
   ^^^^^^^^^^^^
   Design and implementation of the builtin I/O drivers that eases the integration
   of computing engines to existing infrastructure.
