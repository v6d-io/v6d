#!/usr/env/env python3
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
import asyncio
import os
import sys
import time
from enum import Enum
from logging import fatal
from signal import SIGTERM
from time import sleep

import numpy as np
import pandas as pd
import pyarrow as pa

import psutil

import vineyard as vy

socket_name = ""
timestamp = int(time.time())
socket_name = "/var/run/%dvineyard.sock" % (timestamp)
default_cache_size = 1024 * 1024  # unit: byte
test_mount_dir = "/tmp/vyfs-test%d" % (timestamp)


class Type(Enum):
    STRING = 1
    INT64 = 2
    DOUBLE = 3


async def start_vineyard_server():
    print("vineyard started")

    cmd = ["sudo", "-E", "python3", "-m", "vineyard", "--socket=%s" % socket_name]
    cmd = ' '.join(cmd)
    print("initilize vineyard by " + cmd)

    proc = await asyncio.create_subprocess_shell(cmd, stdout=sys.stdout)

    return proc


async def start_fuse_server():
    print("fuse started")
    fuse_bin = [
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            '..',
            '..',
            'build',
            'bin',
            "vineyard-fusermount",
        )
    ]
    fuse_param = [
        "-f",
        "-s",
        "--vineyard-socket=" + socket_name,
        "--max-cache-size=%d" % default_cache_size,
    ]
    os.mkdir(test_mount_dir)
    fuse_dir = [test_mount_dir]
    cmd = fuse_bin + fuse_param + fuse_dir
    cmd = ' '.join(cmd)
    print("initilize fuse by " + cmd)
    proc = await asyncio.create_subprocess_shell(cmd, stdout=sys.stdout)
    return proc


def connect_to_server():
    client = vy.connect(socket_name)
    return client


def interrupt_proc(proc):
    print("interrupt")
    proc.send_signal(SIGTERM)


def generate_dataframe(size=(3, 4)):
    height, width = size
    ldf = pd.DataFrame(
        np.random.randint(0, 100, size=(height, width)) * 2.3,
        columns=[''.join(['a'] * i) for i in range(1, width + 1)],
    )
    rdf = pd.DataFrame(
        np.random.randint(0, 100, size=(height, width)),
        columns=[''.join(['b'] * i) for i in range(1, width + 1)],
    )
    return pd.concat([ldf, rdf], axis=1, join="inner")


def generate_string_array(length=20):
    res = []
    alphabet = [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z',
        ' ',
    ]

    for _ in range(1, length):
        s_length = np.random.randint(1, length)
        res.append(''.join(np.random.choice(alphabet, s_length)))

    return res


def generate_array(type: Type, length=20):
    f = {
        Type.INT64: lambda x: np.random.randint(0, 1000, x),
        Type.DOUBLE: lambda x: np.random.uniform(low=0, high=1000, size=x),
        Type.STRING: generate_string_array,
    }
    return pa.array(f[type](length))


def assert_dataframe(stored_df: pd.DataFrame, extracted_df: pa.Table):
    pdf = pa.Table.from_pandas(stored_df)
    assert extracted_df.equals(pdf), "data frame unmatch"


def assert_array(stored_arr: pa.Array, extracted_array: pa.Array):
    assert stored_arr.equals(extracted_array), "array unmatch"


def read_data_from_fuse(vid):
    with open(os.path.join(test_mount_dir, vid), 'rb') as source:
        with pa.ipc.open_stream(source) as reader:
            data = reader.read_all()
            return data


def test_fuse_array(data, client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28] + ".arrow")
    print("data: ")
    print(data)
    print("extracted data: ")
    extracted_data = extracted_data.column("a").chunk(0)
    print(extracted_data)
    assert_array(data, extracted_data)


def test_fuse_string_array(data, client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28] + ".arrow")
    print("data: ")
    print(data)
    print("extracted data: ")
    extracted_data = extracted_data.column("a").chunk(0)
    print(extracted_data)
    assert compare_two_string_array(data, extracted_data), "string array not the same"


def compare_two_string_array(arr_str_1, arr_str_2):
    a = arr_str_1
    b = arr_str_2
    if len(a) != len(b):
        return False
    else:
        for i, j in zip(a, b):
            if str(i) != str(j):
                print("they are different")
                print(i)
                print(j)
                return False
    return True


def test_fuse_df(data, client):
    id = client.put(data)
    extracted_data = read_data_from_fuse(str(id)[11:28] + ".arrow")
    assert_dataframe(data, extracted_data)


# string_array = generate_array(Type.STRING)


def test_cache_manager(data, client, fuse_process):
    for _ in range(1, 100):
        id = client.put(data)
        _ = read_data_from_fuse(str(id)[11:28] + ".arrow")

    memory_usage_before = psutil.Process(fuse_process.pid).memory_info().rss / 1024**2

    for _ in range(1, 100):
        id = client.put(data)
        _ = read_data_from_fuse(str(id)[11:28] + ".arrow")

    memory_usage_after = psutil.Process(fuse_process.pid).memory_info().rss / 1024**2
    if abs((memory_usage_before) - (memory_usage_after)) > 0.0001:
        fatal("memory usage changed")
    print("cache manager test passed")


if __name__ == "__main__":

    # logger.basicConfig(filename='test.log', level=logger.DEBUG)

    print("started")
    vineyard_server = asyncio.run(start_vineyard_server())
    print(vineyard_server)
    sleep(2)
    fuse_server = asyncio.run(start_fuse_server())
    sleep(2)
    print("server started")
    client = connect_to_server()

    # test array
    int_array = generate_array(Type.INT64)
    test_fuse_array(int_array, client)
    double_array = generate_array(Type.DOUBLE)
    test_fuse_array(double_array, client)

    string_array = generate_array(Type.STRING)
    test_fuse_string_array(string_array, client)
    print("array_test passed")
    # test df
    double_df = generate_dataframe()
    test_fuse_df(double_df, client)
    print("df_test_passed")

    test_cache_manager(double_df, client, fuse_server)

    interrupt_proc(fuse_server)
    interrupt_proc(vineyard_server)
    print("all passed")
