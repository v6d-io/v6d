.. _streams-in-vineyard:

Streams in Vineyard
===================

Stream is an abstraction upon the immutable data sharing storage that allows
convenient pipelining between computing engines. Like
`pipe <https://man7.org/linux/man-pages/man2/pipe.2.html>`_, stream in vineyard
enable efficiently inter-engine communication without introducing the overhead
of data serialization/deserialization and copying.

A common use case of stream in vineyard is one process continues to producing
data chunks (e.g., an IO reader) and another process can do some scan computing
over the data (e.g., filtering and aggregation operators). A stream is consist
of a sequence of immutable data chunks that produced by the former engine and
consumed by the later engine.

This section will cover the usage of streams in vineyard.

Using streams
-------------

We first import required packages:


.. code:: python

    import threading
    import time
    from typing import List

    import numpy as np
    import pandas as pd

    import vineyard
    from vineyard.io.recordbatch import RecordBatchStream

.. tip::

    Vineyard has defined some built-in stream types, e.g.,
    :class:`vineyard.io.recordbatch.RecordBatchStream`. For other stream types,
    you could refer to :ref:`python-api-streams`.

Then we define a producer which generate some random dataframe chunks and put into
the stream:

.. code:: python
   :caption: A producer of :code:`RecordBatchStream`

    def generate_random_dataframe(dtypes, size):
        columns = dict()
        for k, v in dtypes.items():
            columns[k] = np.random.random(size).astype(v)
        return pd.DataFrame(columns)

    def producer(stream: RecordBatchStream, total_chunks, dtypes, produced: List):
        writer = stream.writer
        for idx in range(total_chunks):
            time.sleep(idx)
            chunk = generate_random_dataframe(dtypes, 2)  # np.random.randint(10, 100))
            chunk_id = vineyard_client.put(chunk)
            writer.append(chunk_id)
            produced.append((chunk_id, chunk))
        writer.finish()

And a consumer which takes the chunks from the stream in a loop until receive a
:code:`StopIteration` exception:

.. code:: python
   :caption: A consumer of :code:`RecordBatchStream`

    def consumer(stream: RecordBatchStream, total_chunks, produced: List):
        reader = stream.reader
        index = 0
        while True:
            try:
                chunk = reader.next()
                print('reader receive chunk:', type(chunk), chunk)
                pd.testing.assert_frame_equal(produced[index][1], chunk)
            except StopIteration:
                break
            index += 1

Finally, we can test the producer and consumer using two thread:

.. code:: python
   :caption: Connect the producer and consumer threads using vineyard stream

    def test_recordbatch_stream(vineyard_client, total_chunks):
        stream = RecordBatchStream.new(vineyard_client)
        dtypes = {
            'a': np.dtype('int'),
            'b': np.dtype('float'),
            'c': np.dtype('bool'),
        }

        client1 = vineyard_client.fork()
        client2 = vineyard_client.fork()
        stream1 = client1.get(stream.id)
        stream2 = client2.get(stream.id)

        produced = []

        thread1 = threading.Thread(target=consumer, args=(stream1, total_chunks, produced))
        thread1.start()

        thread2 = threading.Thread(target=producer, args=(stream2, total_chunks, dtypes, produced))
        thread2.start()

        thread1.join()
        thread2.join()

    if __name__ == '__main__':
        vineyard_client = vineyard.connect("/tmp/vineyard.sock")
        test_recordbatch_stream(vineyard_client, total_chunks=10)

For more detailed API about the streams, please refer to :ref:`python-api-streams`.

