vineyard-graph
==============

:code:`vineyard-graph` defines the graph data structures that can be shared
among graph computing engines.

* `vineyard-graph-loader <#vineyard-graph-loader>`_

  * `Usage <#usage>`_

    * `Using Command-line Arguments <#using-command-line-arguments>`_
    * `Using a JSON Configuration <#using-a-json-configuration>`_

  * `References <#references>`_

    * `Vertices <#vertices>`_
    * `Edges <#edges>`_
    * `Data Sources <#data-sources>`_
    * `Read Options <#read-options>`_
    * `Global Options <#global-options>`_

CMake configure options
-----------------------

- :code:`VINEYARD_GRAPH_MAX_LABEL_ID`

  The internal vertex id (aka. :code:`VID`) in vineyard is encoded as fragment id, vertex label id
  and vertex offset. The option :code:`VINEYARD_GRAPH_MAX_LABEL_ID` decides the bit field width of
  label id in :code:`VID`. Decreasing this value can be helpful to support larger number of vertices
  when using :code:`int32_t` as :code:`VID_T`.

  Defaults to `128`, can be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`.

vineyard-graph-loader
---------------------

:code:`vineyard-graph-loader` is a graph loader used to load graphs from
the CSV format into vineyard.

Usage
^^^^^

.. code:: bash

    $ ./vineyard-graph-loader
    Usage: loading vertices and edges as vineyard graph.

           ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] \
                                   <e_label_num> <efiles...> <v_label_num> <vfiles...> \
                                   [directed] [generate_eid] [string_oid]

       or: ./vineyard-graph-loader [--socket <vineyard-ipc-socket>] --config <config.json>

The program :code:`vineyard-graph-loader` first accepts an optional argument
:code:`--socket <vineyard-ipc-socket>` which specifies the IPC docket that the
loader will connected to. If the option is not provided, the loader will try to
resolve the IPC socket from the environment variable `VINEYARD_IPC_SOCKET`.

The graph can be loaded either via command line arguments or a JSON configuration.

Using Command-line Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`vineyard-graph-loader` accepts a sequence of command line arguments to
specify the edge files and vertex files, e.g.,

.. code:: bash

   $ ./vineyard-graph-loader 2 "modern_graph/knows.csv#header_row=true&src_label=person&dst_label=person&label=knows&delimiter=|" \
                               "modern_graph/created.csv#header_row=true&src_label=person&dst_label=software&label=created&delimiter=|" \
                             2 "modern_graph/person.csv#header_row=true&label=person&delimiter=|" \
                               "modern_graph/software.csv#header_row=true&label=software&delimiter=|"

Using a JSON Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`vineyard-graph-loader` can also accept a config file (in JSON format) as well
to specify the vertex files and edge files that would be loaded. as well as global
flags, for example,

.. code:: bash

   $ ./vineyard-graph-loader --config config.json

Here is an example of the `config.json` file for the "modern graph":

.. code:: json

   {
       "vertices": [
           {
               "data_path": "modern_graph/person.csv",
               // can also be absolute path or path with environment variables
               //
               // "data_path": "/datasets/modern_graph/person.csv",
               // "data_path": "$DATASET/modern_graph/person.csv",
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
       "local_vertex_map": 0,
       "print_normalized_schema": 1
   }

References
^^^^^^^^^^

Vertices
~~~~~~~~

Each vertices can have the following configurations:

- :code:`data_path`: the path of the given sources, environment variables are supported,
  e.g., :code:`$HOME/data/person.csv`. See also `Data Sources <#data-sources>`_.
- :code:`label`: the label of the vertex, e.g., :code:`person`.
- :code:`options`: the options used to read the file, e.g., :code:`header_row=true&delimiter=|`.
  The detailed options are listed in `Read Options <#read-options>`_.

Edges
~~~~~

Each edges can have the following configurations:

- :code:`data_path`: the path of the given sources, environment variables are supported,
  e.g., :code:`$HOME/data/knows.csv`. See also `Data Sources <#data-sources>`_.
- :code:`label`: the label of the edge, e.g., :code:`knows`.
- :code:`src_label`: the label of the source vertex, e.g., :code:`person`.
- :code:`dst_label`: the label of the destination vertex, e.g., :code:`person`.
- :code:`options`: the options used to read the file, e.g., :code:`header_row=true&delimiter=|`.
  The detailed options are listed in `Read Options <#read-options>`_.

Data Sources
~~~~~~~~~~~~

The :code:`data_path` can be local files, S3 files, HDFS files, or vineyard streams.

When it comes to local files, it can be a relative path, an absolute path, or a path
with environment variables, e.g.,

- :code:`data/person.csv`
- :code:`/dataset/data/person.csv`
- :code:`$HOME/data/person.csv`

When it comes to S3 files and HDFS files, the support for various sources in :code:`data_path`
can be archived in two approaches:

- Option 1: use `vineyard.io <https://github.com/v6d-io/v6d/tree/main/python/vineyard/drivers/io>`_
  to read the given sources as vineyard streams first, and pass the stream as :code:`vineyard://<object_id_string>`
  as :code:`data_path` to the loader.

- Option 2: configure the arrow dependency that used to build the vineyard-graph-loader to support
  S3 and HDFS with `extra cmake flags <https://arrow.apache.org/docs/developers/cpp/building.html#optional-components>`_.

For edges that have different kinds of :code:`(src, dst)` pair, just repeat the "edge" object in
the configuration file, e.g.,

.. code:: json

   {
       "vertices": [
          ...
       ],
       "edges": [
           {
               "data_path": "person_knows_person.csv",
               "label": "knows",
               "src_label": "person",
               "dst_label": "person",
               "options": "header_row=true&delimiter=|"
           },
           {
               "data_path": "person_knows_item.csv",
               "label": "knows",
               "src_label": "person",
               "dst_label": "item",
               "options": "header_row=true&delimiter=|"
           },
           ...
       ],
      ...
   }

Read Options
~~~~~~~~~~~~

The read options are used to specify how to read the given sources, multiple options
should be separated by :code:`&` or :code:`#`, and are listed as follows:

- :code:`header_row`: whether the first row of CSV file is the header row or not,
  default is :code:`0`.
- :code:`delimiter`: the delimiter of the CSV file, default is :code:`,`.

- :code:`schema`: the columns to specify in the CSV file, default is empty that indicates
  all columns will be included. The :code:`schema` is a :code:`,`-separated list of column names
  or column indices, e.g., :code:`name,age` or :code:`0,1`.
- :code:`column_types`: specify the data type of each column, default is empty that
  indicates the types will be inferred from the data. The `column_types` is a `,`
  separated list of data types, e.g., :code:`string,int64`. **If specified, the types
  of ALL columns must be specified and partial-specification won't work.**

  The supported data types are listed as follows:

  - :code:`bool`: boolean type.
  - :code:`int8_t`, :code:`int8`, :code:`byte`: signed 8-bit integer type.
  - :code:`uint8_t`, :code:`uint8`, :code:`char`: unsigned 8-bit integer type.
  - :code:`int16_t`, :code:`int16`, :code:`half`: signed 16-bit integer type.
  - :code:`uint16_t`, :code:`uint16`: unsigned 16-bit integer type.
  - :code:`int32_t`, :code:`int32`, :code:`int`: signed 32-bit integer type.
  - :code:`uint32_t`, :code:`uint32`: unsigned 32-bit integer type.
  - :code:`int64_t`, :code:`int64`, :code:`long`: signed 64-bit integer type.
  - :code:`uint64_t`, :code:`uint64`: unsigned 64-bit integer type.
  - :code:`float`: 32-bit floating point type.
  - :code:`double`: 64-bit floating point type.
  - :code:`string`, :code:`std::string`, :code:`str`: string type.

- :code:`include_all_columns`: whether to include all columns in the CSV file or not,
  default is :code:`0`. **If specified, the columns that exists in the data file,
  but not be listed in the `schema` option will be read as well.**

  The combination of :code:`schema` and :code:`include_all_columns` is useful for scenarios
  where we need to specify the order the columns that not the same with the content of the
  file, but do not want to tell all column names in detail. For example, if the file contains
  the ID column in the **third** column but we want to use it as the vertices IDs, we
  could have :code:`schema=2&include_all_columns=1` the all columns will be read, but the
  **third** column in the file will be placed at the **first** column in the result table.

Global Options
~~~~~~~~~~~~~~

Global options controls how the fragment is constructed from given vertices
and edges and are listed as follows:

- :code:`directed`: whether the graph is directed or not, default is :code:`1`.
- :code:`generate_eid`: whether to generate edge id or not, default is :code:`0`. **Generating
  edge id is usually required in GraphScope GIE.**
- :code:`retain_oid`: whether to retain the original ID of the vertex's property table or not,
  default is :code:`0`. **Retaining original ID in vertex's property table is usually required
  in GraphScope GIE.**
- :code:`oid_type`: the type of the original ID of the vertices, default is :code:`int64_t`.
  Can be :code:`int64_t` and :code:`string`.

- :code:`large_vid`: whether the vertex id is large or not, default is :code:`1`. If you are
  sure that the number of vertices is fairly small (:code:`< 2^(31-log2(vertex_label_number)-1)`),
  setting :code:`large_vid` to :code:`0` can reduce the memory usage. **Note that
  :code:`large_vid=0` isn't compatible with GraphScope GIE.**
- :code:`local_vertex_map`: whether to use local vertex map or not, default is :code:`0`.
  Using local vertex map is usually helpful to reduce the memory usage. **Note that
  :code:`local_vertex_map=0` isn't compatible with GraphScope GIE.**

- :code:`print_memory_usage`: whether to print the memory usage of the graph to :code:`STDERR`
  or not. Default is :code:`0`.
- :code:`print_normalized_schema`: whether to print the **normalized** schema of the graph to
  :code:`STDERR` or not, default is :code:`0`. The word "normalized" means make the same property
  name has the same property id across different labels, **which is required by GraphScope GIE.**

- :code:`dump`: a string that indicates a directory to dump the graph to, default is empty that
  indicates no dump, e.g., :code:`"dump": "/tmp/dump-graph"`.
- :code:`dump_dry_run_rounds`: if greater than :code:`0`, will traverse the graph for
  :code:`dump_dry_run_rounds` times to measure the edge (CSR) accessing performance. Default
  is :code:`0`.
- :code:`use_perfect_hash` whether to use perfect map when construct vertex map. Default is
  :code:`0`. Using perfect map is usually helpful to reduce the memory usage. But it is not
  recommended when the graph is small.
