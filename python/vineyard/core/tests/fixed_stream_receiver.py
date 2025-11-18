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

from datetime import datetime
import mmap
import sys
import time
import vineyard
from vineyard.io.fixed_blob import FixedBlobStream

from vineyard._C import ObjectID

blob_num = 10
blob_size = 1024 * 1024 * 2

def run_receiver(client: vineyard.Client, mm: mmap.mmap, ipc_socket: str, rpc_endpoint: str):
  fixed_blob_stream = FixedBlobStream.new(client, "test-stream-5", blob_num, blob_size, True, rpc_endpoint)
  stream_reader = fixed_blob_stream.open_reader(client, True, 10000)
  offset_list = []
  for i in range(blob_num):
    offset_list.append(i * blob_size)
  
  stream_reader.activate_stream_with_offset(offset_list)

  total_finished = stream_reader.check_block_received(-1)
  print("Stream is :", "finished" if total_finished else "not finished")
  
  for i in range(blob_num):
    finished = False
    while not finished:
      start_time = datetime.now().microsecond
      try:
        finished = stream_reader.check_block_received(i)
      except Exception as e:
        print(f"Error checking block {i}: {e}")
        break

      end_time = datetime.now().microsecond
      print(f"Waiting for chunk {i}...")
      time.sleep(0.2)
    
    if finished is not True:
      while True:
        aborted = stream_reader.abort()
        if aborted:
          print("Stream aborted, bye...")
          return

    for j in range(blob_size):
      assert mm.read_byte() == j % 256
    print("Chunk ", i, " received successfully")

  for i in range(blob_num):
    finished = False
    while not finished:
      start_time = datetime.now().microsecond
      finished = stream_reader.check_block_received(i)
      end_time = datetime.now().microsecond
      print(f"check used time: {end_time - start_time} us")

  start_time = datetime.now().microsecond
  total_finished = stream_reader.check_block_received(-1)
  end_time = datetime.now().microsecond
  print("Stream is :", "finished" if total_finished else "not finished")
  print("check all use time: ", end_time - start_time, " us")
  stream_reader.finish_and_delete()


def __main__():
  arguments = sys.argv[1:]
  if len(arguments) < 2:
    print("Usage: fixed_stream_receiver.py <ipc_socket> <rpc_endpoint>")
    return 1
  
  ipc_socket = arguments[0]
  rpc_endpoint = arguments[1]
  client = vineyard.connect(ipc_socket)
  client.timeout_seconds = 5

  list = client.get_vineyard_mmap_fd()
  fd = list[0]
  offset = list[2]

  mm = mmap.mmap(fd, 0)
  mm.seek(offset)

  run_receiver(client, mm, ipc_socket, rpc_endpoint)

if __name__ == "__main__":
  __main__()