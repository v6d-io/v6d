#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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
import mmap
import sys
import threading
from time import sleep

import vineyard
from vineyard.io.fixed_blob import FixedBlobStream

blob_num = 10
blob_size = 1024 * 1024 * 2


def check_received(
    client: vineyard.Client,
    stream_id: vineyard.ObjectID,
    stream_reader: FixedBlobStream.Reader,
):
    finished = False
    while not finished:
        finished = stream_reader.check_block_received(-1)
        print("Waiting for stream to finish...")
        sleep(2)

    success = False
    success = stream_reader.abort()
    print("Stream aborted: ", success)
    stream_reader.finish_and_delete()


def run_sender(client: vineyard.Client, mm: mmap.mmap):
    fixed_blob_stream = FixedBlobStream.new(
        client, "test-stream-5", blob_num, blob_size, False, ""
    )
    stream_writer = fixed_blob_stream.open_writer(client)

    offset_list = []
    for i in range(blob_num):
        for j in range(blob_size):
            mm.write_byte(j % 256)

    for i in range(blob_num):
        offset_list.append(i * blob_size)

    thread = threading.Thread(
        target=check_received,
        args=(
            client,
            id,
            stream_writer,
        ),
    )

    thread.start()

    for offset in offset_list:
        stream_writer.append(offset)
        sleep(1)

    thread.join()


def __main__():
    arguments = sys.argv[1:]
    if len(arguments) < 1:
        print("Usage: fixed_stream_receiver.py <ipc_socket>")
        return 1

    ipc_socket = arguments[0]
    client = vineyard.connect(ipc_socket)
    client.timeout_seconds = 5

    list = client.get_vineyard_mmap_fd()
    fd = list[0]
    offset = list[2]

    mm = mmap.mmap(fd, 0)
    mm.seek(offset)

    run_sender(client, mm)


if __name__ == "__main__":
    __main__()
