IO Adaptors
-----------

Vineyard has a set of prebuilt IO adaptors, that can serve as common routines for
various IO operations and can take place of boilerplate parts in computation tasks.

Their functionality are described as follows:

+ :code:`read_local_bytes`

  .. code:: console

    Usage: vineyard_read_local_bytes <ipc_socket> <efile> <proc_num> <proc_index>

  Read a local file to :class:`ByteStream`.

+ :code:`read_local_orc`

  .. code:: console

    Usage: vineyard_read_local_orc <ipc_socket> <orc file path> <proc num> <proc index>

  Read a local ORC file to :class:`DataframeStream`.

+ :code:`read_kafka_bytes`

  .. code:: console

    Usage: vineyard_read_kafka_bytes <ipc_socket> <kafka_address> <proc_num> <proc_index>

  Read a kafka stream to :class:`ByteStream`.

+ :code:`read_hdfs_bytes`

  .. code:: console

    Usage: vineyard_read_hdfs_bytes <ipc_socket> <efile> <proc_num> <proc_index>

  Read a HDFS file to :class:`ByteStream`.

+ :code:`read_hdfs_orc`

  .. code:: console

    Usage: vineyard_read_hdfs_orc <ipc_socket> <efile> <proc_num> <proc_index>

  Read a HDFS ORC file to :class:`DataframeStream`.

+ :code:`read_hive_orc`

  .. code:: console

    Usage: vineyard_read_hive_orc <ipc_socket> <efile> <proc_num> <proc_index>

  Read a Hive table (on HDFS and in ORC format) to :class:`DataframeStream`.

+ :code:`write_local_dataframe`

  .. code:: console

    Usage: vineyard_write_local_dataframe <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a dataframe stream to a local file.

+ :code:`write_local_orc`

  .. code:: console

    Usage: vineyard_write_local_orc <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a dataframe stream to a local ORC file.

+ :code:`write_kafka_bytes`

  .. code:: console

    Usage: vineyard_write_kafka_bytes <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a byte stream to a kafka stream.

+ :code:`write_kafka_dataframe`

  .. code:: console

    Usage: vineyard_write_kafka_dataframe <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a dataframe stream to a kafka stream.

+ :code:`write_hdfs_bytes`

  .. code:: console

    Usage: vineyard_write_hdfs_bytes <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a byte stream to a HDFS.

+ :code:`write_hdfs_bytes`

  .. code:: console

    Usage: vineyard_write_hdfs_bytes <ipc_socket> <stream_id> <ofile> <proc_num> <proc_index>

  Write a dataframe stream to a HDFS in ORC format.

+ :code:`read_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_read_vineyard_dataframe <ipc_socket> <vineyard_address> <proc_num> <proc_index>

  Read a vineyard global dataframe to a dataframe stream

+ :code:`write_vineyard_dataframe`

  .. code:: console

    Usage: vineyard_write_vineyard_dataframe <ipc_socket> <stream_id> <proc_num> <proc_index>

  Write a dataframe stream to a series of vineyard dataframes
