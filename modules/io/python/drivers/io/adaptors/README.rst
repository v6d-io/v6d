IO Adaptors
-----------

Vineyard has a set of prebuilt IO adaptors, that can serve as common routines for
various IO operations and can take place of boilerplate parts in computation tasks.

Vineyard is capable of reading from and writing data to multiple file systems.
Behind the scene, it leverage `fsspec` to delegate the workload to various file system implementations.

Specifically, we can specify parameters to be passed to the file system, through the `storage_options` parameter.
`storage_options` is a dict that pass additional keywords to the file system,
For instance, we could combine `path` = `hdfs:///path/to/file` with `storage_options` = `{"host": "localhost", "port": 9600}`
to read from a HDFS.

Alternatively, we can encode such information into the path by using methods,
such as: `hdfs://<ip>:<port>/path/to/file`.

To read from multiple files you can pass a globstring or a list of paths,
with the caveat that they must all have the same protocol.

Their functionality are described as follows:

+ :code:`read_bytes`

  .. code:: console

    Usage: vineyard_read_bytes <ipc_socket> <path> <storage_options> <read_options> <proc_num> <proc_index>

  Read a file on local file systems, OSS, HDFS, S3, etc. to :class:`ByteStream`.

+ :code:`write_bytes`

  .. code:: console

    Usage: vineyard_write_bytes <ipc_socket> <path> <stream_id> <storage_options> <write_options> <proc_num> <proc_index>

  Write a :class:`ByteStream` to a file on local file system, OSS, HDFS, S3, etc.

+ :code:`read_orc`

  .. code:: console

    Usage: vineyard_read_orc <ipc_socket> <path/directory> <storage_options> <read_options> <proc_num> <proc_index>

  Read a ORC file on local file systems, OSS, HDFS, S3, etc. to :class:`DataframeStream`.

+ :code:`write_orc`

  .. code:: console

    Usage: vineyard_read_orc <ipc_socket> <path/directory> <storage_options> <read_options> <proc_num> <proc_index>

  Write a :class:`DataframeStream` to a ORC file on local file system, OSS, HDFS, S3, etc.

+ :code:`read_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_read_vineyard_dataframe <ipc_socket> <vineyard_address> <storage_options> <read_options> <proc num> <proc index>

  Read a :class:`DataFrame` in vineyard as a :class:`DataframeStream`.

+ :code:`write_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_write_vineyard_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>

  Write a :class:`DataframeStream` to a :class:`DataFrame` in vineyard.

+ :code:`serializer`

  .. code:: console

    Usage: vineyard_serializer <ipc_socket> <object_id>

  Serialize a vineyard object (non-global or global) as a :class:`ByteStream` or a set of :class:`ByteStream` (:class:`StreamCollection`).

+ :code:`deserializer`

  .. code:: console

    Usage: vineyard_deserializer <ipc_socket> <object_id>

  Deserialize a :class:`ByteStream` or a set of :class:`ByteStream` (:class:`StreamCollection`) as a vineyard object.

+ :code:`read_bytes_collection`

  .. code:: console

    Usage: vineyard_read_bytes_collection <ipc_socket> <prefix> <storage_options> <proc_num> <proc_index>

  Read a directory (on local filesystem, OSS, HDFS, S3, etc.) as a :class:`ByteStream` or a set of :class:`ByteStream` (:class:`StreamCollection`).

+ :code:`write_bytes_collection`

  .. code:: console

    Usage: vineyard_write_vineyard_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>

  Write a :class:`ByteStream` or a set of :class:`ByteStream` (:class:`StreamCollection`) to a directory (on local filesystem, OSS, HDFS, S3, etc.).

+ :code:`parse_bytes_to_dataframe`

  .. code:: console

    Usage: vineyard_parse_bytes_to_dataframe.py <ipc_socket> <stream_id> <proc_num> <proc_index>

  Parse a :class:`ByteStream` (in CSV format) as a :class:`DataframeStream`.

+ :code:`parse_dataframe_to_bytes`

  .. code:: console

    Usage: vineyard_parse_dataframe_to_bytes <ipc_socket> <stream_id> <proc_num> <proc_index>

  Serialize a :class:`DataframeStream` to a :class:`ByteStream` (in CSV format).

+ :code:`dump_dataframe`

  .. code:: console

    Usage: vineyard_dump_dataframe <ipc_socket> <stream_id>

  Dump the content of a :class:`DataframeStream`, for debugging usage.
