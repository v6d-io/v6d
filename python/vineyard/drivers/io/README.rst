.. image:: https://v6d.io/_static/vineyard_logo.png
   :target: https://v6d.io
   :align: center
   :alt: vineyard
   :width: 397px

vineyard-io: IO drivers for `vineyard <https://v6d.io>`_
--------------------------------------------------------

vineyard-io is a collection of IO drivers for `vineyard <https://v6d.io>`_. Currently it supports

* Local filesystem
* AWS S3
* Aliyun OSS
* Hadoop filesystem

The vineyard-io package leverages the `filesystem-spec <http://filesystem-spec.readthedocs.io/>`_
to support other storage sinks and sources in a unified fashion. Other adaptors that works for fsspec
could be plugged in as well.

IO Adaptors
~~~~~~~~~~~

Vineyard has a set of prebuilt IO adaptors, that can serve as common routines for
various IO operations and can take place of boilerplate parts in computation tasks.

Vineyard is capable of reading from and writing data to multiple file systems.
Behind the scene, it leverage :code:`fsspec` to delegate the workload to various file system implementations.

Specifically, we can specify parameters to be passed to the file system, through the :code:`storage_options` parameter.
:code:`storage_options` is a dict that pass additional keywords to the file system,
For instance, we could combine :code:`path` = `hdfs:///path/to/file` with :code:`storage_options = {"host": "localhost", "port": 9600}`
to read from a HDFS.

Note that you must encode the :code:`storage_options` by base64 before passing it to the scripts.

Alternatively, we can encode such information into the path,
such as: :code:`hdfs://<ip>:<port>/path/to/file`.

To read from multiple files you can pass a glob string or a list of paths,
with the caveat that they must all have the same protocol.

Their functionality are described as follows:

+ :code:`read_bytes`

  .. code:: console

    Usage: vineyard_read_bytes <ipc_socket> <path> <storage_options> <read_options> <proc_num> <proc_index>

  Read a file on local file systems, OSS, HDFS, S3, etc. to :code:`ByteStream`.

+ :code:`write_bytes`

  .. code:: console

    Usage: vineyard_write_bytes <ipc_socket> <path> <stream_id> <storage_options> <write_options> <proc_num> <proc_index>

  Write a :code:`ByteStream` to a file on local file system, OSS, HDFS, S3, etc.

+ :code:`read_orc`

  .. code:: console

    Usage: vineyard_read_orc <ipc_socket> <path/directory> <storage_options> <read_options> <proc_num> <proc_index>

  Read a ORC file on local file systems, OSS, HDFS, S3, etc. to :code:`DataframeStream`.

+ :code:`write_orc`

  .. code:: console

    Usage: vineyard_read_orc <ipc_socket> <path/directory> <storage_options> <read_options> <proc_num> <proc_index>

  Write a :code:`DataframeStream` to a ORC file on local file system, OSS, HDFS, S3, etc.

+ :code:`read_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_read_vineyard_dataframe <ipc_socket> <vineyard_address> <storage_options> <read_options> <proc num> <proc index>

  Read a :code:`DataFrame` in vineyard as a :code:`DataframeStream`.

+ :code:`write_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_write_vineyard_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>

  Write a :code:`DataframeStream` to a :code:`DataFrame` in vineyard.

+ :code:`serializer`

  .. code:: console

    Usage: vineyard_serializer <ipc_socket> <object_id>

  Serialize a vineyard object (non-global or global) as a :code:`ByteStream` or a set of :code:`ByteStream` (:code:`StreamCollection`).

+ :code:`deserializer`

  .. code:: console

    Usage: vineyard_deserializer <ipc_socket> <object_id>

  Deserialize a :code:`ByteStream` or a set of :code:`ByteStream` (:code:`StreamCollection`) as a vineyard object.

+ :code:`read_bytes_collection`

  .. code:: console

    Usage: vineyard_read_bytes_collection <ipc_socket> <prefix> <storage_options> <proc_num> <proc_index>

  Read a directory (on local filesystem, OSS, HDFS, S3, etc.) as a :code:`ByteStream` or a set of :code:`ByteStream` (:code:`StreamCollection`).

+ :code:`write_bytes_collection`

  .. code:: console

    Usage: vineyard_write_vineyard_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>

  Write a :code:`ByteStream` or a set of :code:`ByteStream` (:code:`StreamCollection`) to a directory (on local filesystem, OSS, HDFS, S3, etc.).

+ :code:`parse_bytes_to_dataframe`

  .. code:: console

    Usage: vineyard_parse_bytes_to_dataframe.py <ipc_socket> <stream_id> <proc_num> <proc_index>

  Parse a :code:`ByteStream` (in CSV format) as a :code:`DataframeStream`.

+ :code:`parse_dataframe_to_bytes`

  .. code:: console

    Usage: vineyard_parse_dataframe_to_bytes <ipc_socket> <stream_id> <proc_num> <proc_index>

  Serialize a :code:`DataframeStream` to a :code:`ByteStream` (in CSV format).

+ :code:`dump_dataframe`

  .. code:: console

    Usage: vineyard_dump_dataframe <ipc_socket> <stream_id>

  Dump the content of a :code:`DataframeStream`, for debugging usage.
