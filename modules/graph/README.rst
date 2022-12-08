vineyard-graph
==============

:code:`vineyard-graph` defines the graph data structures that can be shared
among graph computing engines.

vineyard-graph-loader
---------------------

:code:`vineyard-graph-loader` is a graph loader that can be used to loading
graphs from the CSV format into vineyard.

Usage
^^^^^

.. code:: bash

    $ ./vineyard-graph-loader
    Usage: loading vertices and edges as vineyard graph.

           ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] \
                                   <e_label_num> <efiles...> <v_label_num> <vfiles...> \
                                   [directed] [generate_eid] [string_oid]

       or: ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] --config <config.json>

The program :code:`vineyard-graph-loader` first accepts an option argument :code:`--socket <vineyard-ipc-socket>`
which points the IPC docket that the loader will connected to. If the option is not provided, the loader will
try to resolve the IPC socket from environment variable `VINEYARD_IPC_SOCKET`.

The graph can be loaded from the following two approaches:

- using command line arguments

  The :code:`vineyard-graph-loader` accepts a sequence of command line arguments to specify the edge files
  and vertex files, e.g.,

  .. code:: bash

     $ ./vineyard-graph-loader 2 "modern_graph/knows.csv#header_row=true&src_label=person&dst_label=person&label=knows&delimiter=|" \
                                 "modern_graph/created.csv#header_row=true&src_label=person&dst_label=software&label=created&delimiter=|" \
                               2 "modern_graph/person.csv#header_row=true&label=person&delimiter=|" \
                                 "modern_graph/software.csv#header_row=true&label=software&delimiter=|"

- using a config file

  The :code:`vineyard-graph-loader` can accept a config file (in JSON format) as well to tell the edge files
  and vertex files that would be loaded, e.g.,

  .. code:: bash

     $ ./vineyard-graph-loader --config config.json

  The config file could be (using the "modern graph" as an example):

  .. code:: json

     {
         "vertices": [
             {
                 "data_path": "modern_graph/person.csv",
                 "label": "person",
                 "options": "header_row=true&delimiter=|"
             },
             {
                 "data_path": "modern_graph/software.csv",
                 "label": "software",
                 "options": "header_row=true&delimiter=|"
             }
         ],
         "edges": [
             {
                 "data_path": "modern_graph/knows.csv",
                 "label": "knows",
                 "src_label": "person",
                 "dst_label": "person",
                 "options": "header_row=true&delimiter=|"
             },
             {
                 "data_path": "modern_graph/created.csv",
                 "label": "created",
                 "src_label": "person",
                 "dst_label": "software",
                 "options": "header_row=true&delimiter=|"
             }
         ],
         "directed": 1,
         "generate_eid": 1,
         "string_oid": 0,
         "local_vertex_map": 0
     }
