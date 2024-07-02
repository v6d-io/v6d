.. _streams-in-vineyard:

Streams in Vineyard
===================

Streams in Vineyard serve as an abstraction over the immutable data sharing storage,
facilitating seamless pipelining between computing engines. Similar to
`pipe <https://man7.org/linux/man-pages/man2/pipe.2.html>`_, Vineyard's streams enable
efficient inter-engine communication while minimizing the overhead associated with
data serialization/deserialization and copying.

A typical use case for streams in Vineyard involves one process continuously producing
data chunks (e.g., an IO reader) while another process performs scan computations
on the data (e.g., filtering and aggregation operations). A stream consists of a
sequence of immutable data chunks, produced by the former engine and consumed by the
latter engine.

This section will explore the utilization of streams in Vineyard.

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

Producer and consumer
---------------------

We define a producer that generates random dataframe chunks and inserts them
into the stream:

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

Additionally, we create a consumer that retrieves the chunks from the stream in a
loop, continuing until it encounters a :code:`StopIteration` exception:

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

Streams between processes
-------------------------

Finally, we can test the producer and consumer using two threads:

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
