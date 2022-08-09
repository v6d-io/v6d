#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import json
import logging
import multiprocessing
import os
import sys
from queue import Empty as QueueEmptyException
from queue import Queue as ConcurrentQueue
from typing import Dict
from typing import Tuple  # pylint: disable=unused-import

import vineyard
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard.io.byte import ByteStream
from vineyard.io.stream import StreamCollection
from vineyard.io.utils import BaseStreamExecutor
from vineyard.io.utils import ThreadStreamExecutor
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

logger = logging.getLogger('vineyard')

CHUNK_SIZE = 1024 * 1024 * 128


def build_a_stream(
    client, meta: ObjectMeta, length, path: str, serialization_options: Dict[str, str]
):
    assert meta.typename == 'vineyard::Blob'
    logger.info('creating stream for blob %r at path %s...', meta.id, path)
    params = {
        StreamCollection.KEY_OF_PATH: path,
        'length': length,
        StreamCollection.KEY_OF_OPTIONS: json.dumps(serialization_options),
    }
    return ByteStream.new(client, params)


def serialize_blob_to_stream(
    stream: ByteStream, blob: memoryview, chunk_size: int = CHUNK_SIZE
):
    logger.info('starting processing blob at %s', stream.params.get('path', 'UNKNOWN'))
    total_size, offset = len(blob), 0
    writer: ByteStream.Writer = stream.open_writer()
    while offset < total_size:
        current_chunk_size = min(chunk_size, total_size - offset)
        if current_chunk_size > 0:
            chunk = writer.next(current_chunk_size)
            vineyard.memory_copy(chunk, 0, blob[offset : offset + current_chunk_size])
            offset += current_chunk_size
    logger.info('finished processing blob at %s', stream.params.get('path', 'UNKNOWN'))
    writer.finish()


class SerializeExecutor(BaseStreamExecutor):
    def __init__(
        self,
        task_queue: "ConcurrentQueue[Tuple[ByteStream, memoryview]]",
        chunk_size: int = CHUNK_SIZE,
        serialization_options: Dict[str, str] = None,
    ):
        self._task_queue = task_queue
        self._chunk_size = chunk_size
        self._serialization_options = serialization_options
        if self._serialization_options is None:
            self._serialization_options = dict()

    def compress_chunk(self, blob: memoryview) -> memoryview:
        if len(blob) == 0:
            return blob

        method = self._serialization_options.get('compression_method', None)
        level = self._serialization_options.get('compression_level', None)
        if method == 'zstd':
            import zstd  # pylint: disable=import-outside-toplevel

            if level is None:
                return zstd.compress(bytes(blob))
            else:
                return zstd.compress(bytes(blob), level)

        # no action
        return blob

    def execute(self):
        processed_blobs, processed_bytes = 0, 0
        while True:
            try:
                s, buffer = self._task_queue.get(block=False)
            except QueueEmptyException:
                break
            buffer = self.compress_chunk(buffer)
            processed_blobs += 1
            processed_bytes += len(buffer)
            serialize_blob_to_stream(s, buffer, self._chunk_size)
        return processed_bytes, processed_blobs


def traverse_to_serialize(
    client,
    meta: ObjectMeta,
    queue: "ConcurrentQueue[Tuple[ByteStream, memoryview]]",
    path: str,
    serialization_options: Dict[str, str],
) -> ObjectID:
    """Returns:
    The generated stream or stream collection id.
    """
    if meta.typename == 'vineyard::Blob':
        blob = meta.get_buffer(meta.id)
        s = build_a_stream(
            client, meta, len(blob), os.path.join(path, 'blob'), serialization_options
        )
        queue.put((s, blob))
        return s.id
    else:
        metadata, streams = dict(), []
        metadata[StreamCollection.KEY_OF_GLOBAL] = meta.isglobal
        for k, v in meta.items():
            if k == 'typename':
                metadata['__typename'] = v
            elif isinstance(v, ObjectMeta):
                if v.islocal:
                    streams.append(
                        traverse_to_serialize(
                            client,
                            v,
                            queue,
                            os.path.join(path, k),
                            serialization_options,
                        )
                    )
            else:
                metadata[k] = v
        metadata[StreamCollection.KEY_OF_PATH] = path
        collection = StreamCollection.new(client, metadata, streams)
        return collection.id


def serialize(vineyard_socket, object_id, serialization_options):
    """Serialize a vineyard object as a stream.

    The serialization executes in the following steps:

    1. glob all blobs in the meta

    2. build a stream for each blob

    3. generate a hierarchical `StreamCollection` object as the result
    """
    client = vineyard.connect(vineyard_socket)
    meta = client.get_meta(object_id)

    queue: "ConcurrentQueue[Tuple[ByteStream, memoryview]]" = ConcurrentQueue()
    serialized_id = traverse_to_serialize(
        client, meta, queue, '', serialization_options
    )

    # object id done
    client.persist(serialized_id)
    report_success(serialized_id)

    # start transfer
    #
    # easy to be implemented as a threaded executor in a future
    executor = ThreadStreamExecutor(
        SerializeExecutor,
        parallism=multiprocessing.cpu_count(),
        task_queue=queue,
        serialization_options=serialization_options,
    )
    results = executor.execute()
    logger.info('finish serialization: %s', results)


def main():
    if len(sys.argv) < 3:
        print("usage: ./serializer <ipc_socket> <object_id> [<serialization_options>]")
        sys.exit(1)
    ipc_socket = sys.argv[1]
    object_id = vineyard.ObjectID(sys.argv[2])
    if len(sys.argv) >= 4:
        serialization_options = json.loads(
            base64.b64decode(sys.argv[3].encode("utf-8")).decode("utf-8")
        )
    else:
        serialization_options = {}
    if 'compression_method' not in serialization_options:
        serialization_options['compression_method'] = 'zstd'
    try:
        serialize(ipc_socket, object_id, serialization_options)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        sys.exit(-1)


if __name__ == "__main__":
    main()
