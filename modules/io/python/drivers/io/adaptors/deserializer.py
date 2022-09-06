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

import io
import json
import logging
import multiprocessing
import os
import sys
from queue import Empty as QueueEmptyException
from queue import Queue as ConcurrentQueue
from typing import Dict
from typing import Tuple  # pylint: disable=unused-import
from typing import Union  # pylint: disable=unused-import

import vineyard
from vineyard._C import Blob
from vineyard._C import BlobBuilder
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard.io.byte import ByteStream
from vineyard.io.stream import StreamCollection
from vineyard.io.utils import BaseStreamExecutor
from vineyard.io.utils import ThreadStreamExecutor
from vineyard.io.utils import report_error
from vineyard.io.utils import report_exception
from vineyard.io.utils import report_success

logger = logging.getLogger('vineyard')


def decompress_chunk(blob: memoryview, serialization_options: Dict[str, str]):
    if len(blob) == 0:
        return blob

    if serialization_options:
        method = serialization_options.get('compression_method', None)
        if method == 'zstd':
            import zstd  # pylint: disable=import-outside-toplevel

            return zstd.decompress(bytes(blob))

    # no action
    return blob


def copy_bytestream_to_blob(client, bs: ByteStream, blob: BlobBuilder):
    logger.info(
        "start copying byte stream to blob: %s", bs.params[StreamCollection.KEY_OF_PATH]
    )
    serialization_options = bs.params.get(StreamCollection.KEY_OF_OPTIONS, None)
    if serialization_options is not None:
        serialization_options = json.loads(serialization_options)
    offset = 0
    reader = bs.open_reader(client)
    buffer = blob.buffer
    raw_buffer = io.BytesIO()
    while True:
        try:
            chunk = reader.next()
        except (StopIteration, vineyard.StreamDrainedException):
            break
        raw_buffer.write(chunk)
    blob_data = decompress_chunk(raw_buffer.getbuffer(), serialization_options)
    assert len(blob_data) <= len(
        buffer
    ), "The source and destination buffer sizes don't match: %d vs. %d" % (
        len(blob_data),
        len(buffer),
    )
    if len(blob_data) > 0:
        vineyard.memory_copy(buffer, offset, blob_data)
        offset += len(blob_data)
    return blob.seal(client)


class ReconstructExecututor(BaseStreamExecutor):
    def __init__(
        self,
        client,
        task_queue: "ConcurrentQueue[Tuple[ByteStream, str, Union[BlobBuilder, Blob]]]",
        result_queue: "ConcurrentQueue[Tuple[ObjectID, Blob]]",
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._result_queue = result_queue

    def execute(self):
        processed_blobs, processed_bytes = 0, 0
        while True:
            try:
                bs, blob = self._task_queue.get(block=False)
            except QueueEmptyException:
                break
            # path pattern is: xx/xxx/xx/../xx/blob
            memberpath = bs.params[StreamCollection.KEY_OF_PATH].split(os.path.sep)[-2]
            if isinstance(blob, BlobBuilder):
                self._result_queue.put(
                    (bs.id, memberpath, copy_bytestream_to_blob(self._client, bs, blob))
                )
            else:
                self._result_queue.put((bs.id, memberpath, blob))
            processed_blobs += 1
            processed_bytes += bs.params['length']
        return processed_blobs, processed_bytes


def traverse_to_prepare(
    client,
    stream_id: ObjectID,
    queue: "ConcurrentQueue[Tuple[ByteStream, Union[BlobBuilder, Blob]]]",
):
    stream = client.get(stream_id)
    if isinstance(stream, StreamCollection):
        for s in stream.streams:
            traverse_to_prepare(client, s, queue)
    else:
        if stream.params['length'] == 0:
            builder = client.create_empty_blob()
        else:
            builder = client.create_blob(stream.params['length'])
        queue.put((stream, builder))


def traverse_to_rebuild(client, stream_id: ObjectID, blobs: Dict[ObjectID, Blob]):
    stream = client.get(stream_id)
    if isinstance(stream, StreamCollection):
        fullpath = stream.meta[StreamCollection.KEY_OF_PATH]
        if fullpath:
            memberpath = fullpath.split(os.path.sep)[-1]
        else:
            memberpath = ''
        logger.info('rebuilding object %s as %s', fullpath, memberpath)
        meta = ObjectMeta()
        for k, v in stream.meta.items():
            # erase existing identifiers
            if k in [
                'typename',
                'id',
                'signature',
                'instance_id',
                'transient',
                StreamCollection.KEY_OF_GLOBAL,
                StreamCollection.KEY_OF_PATH,
                StreamCollection.KEY_OF_STREAMS,
            ]:
                continue
            if k == '__typename':
                meta['typename'] = v
            else:
                meta[k] = v
        isglobal = stream.meta[StreamCollection.KEY_OF_GLOBAL]
        if isglobal:
            meta.set_global(True)
        for s in stream.streams:
            name, member = traverse_to_rebuild(client, s, blobs)
            if isglobal:
                client.persist(member.id)
            meta.add_member(name, member)
        meta = client.create_metadata(meta)
        return memberpath, meta
    else:
        memberpath, blob = blobs[stream.id]
        return memberpath, blob


def deserialize(vineyard_socket, object_id, proc_num, proc_index):
    client = vineyard.connect(vineyard_socket)
    streams = client.get(object_id)

    if len(streams) != proc_num:
        report_error("Expected: %s stream partitions" % proc_num)
        sys.exit(-1)

    queue: "ConcurrentQueue[Tuple[ByteStream, Union[BlobBuilder, Blob]]]" = (
        ConcurrentQueue()
    )
    traverse_to_prepare(client, streams[proc_index].id, queue)

    # serve as a stream id -> blob id mapping
    rqueue: "ConcurrentQueue[Tuple[ObjectID, str, Blob]]" = ConcurrentQueue()

    # copy blobs
    executor = ThreadStreamExecutor(
        ReconstructExecututor,
        parallism=multiprocessing.cpu_count(),
        client=client,
        task_queue=queue,
        result_queue=rqueue,
    )
    executor.execute()

    blobs: Dict[ObjectID, Blob] = dict()
    while not rqueue.empty():
        bs, memberpath, blob = rqueue.get(block=False)
        blobs[bs] = (memberpath, blob)

    _, result = traverse_to_rebuild(client, streams[proc_index].id, blobs)
    client.persist(result.id)
    report_success(result.id)


def main():
    if len(sys.argv) < 5:
        print("usage: ./deserializer <ipc_socket> <object_id> <proc_num> <proc_index>")
        sys.exit(1)
    ipc_socket = sys.argv[1]
    object_id = vineyard.ObjectID(sys.argv[2])
    proc_num = int(sys.argv[3])
    proc_index = int(sys.argv[4])
    try:
        deserialize(ipc_socket, object_id, proc_num, proc_index)
    except Exception:  # pylint: disable=broad-except
        report_exception()
        sys.exit(-1)


if __name__ == "__main__":
    main()
