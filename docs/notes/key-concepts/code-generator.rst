Code Generation for Boilerplate
===============================

Sharing objects between engines consists of two basic steps, defining the
data structure and defining the protocol of

Vineyard has already support a set of efficient builtin data types in
the C++ SDK, e.g., :code:`Vector`, :code:`HashMap`, :code:`Tensor`,
:code:`DataFrame`, :code:`Table` and :code:`Graph`, (see :ref:`cpp-api`).
However there are still scenarios where users need to develop their
own data structures and efficiently share the data with Vineyard. Custom
C++ data types could be easily added by following this step-by-step tutorial.

    Note that this tutorial includes code that could be auto-generated for
    keeping clear about the design internals and helping developers get a whole
    picture about how vineyard client works.

