#!/usr/env/env python3
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
import os
from enum import Enum

import numpy as np
import pandas as pd
import pyarrow as pa


class Type(Enum):
    STRING = 1
    INT64 = 2
    DOUBLE = 3


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


def read_data_from_fuse(vid, test_mount_dir):
    with open(os.path.join(test_mount_dir, vid), 'rb') as source:
        with pa.ipc.open_file(source) as reader:
            data = reader.read_all()
            return data


def compare_two_string_array(arr_str_1, arr_str_2):
    a = arr_str_1
    b = arr_str_2
    if len(a) != len(b):
        return False
    else:
        for i, j in zip(a, b):
            if str(i) != str(j):
                return False
    return True


def test_fuse_int64_array(vineyard_client, vineyard_fuse_mount_dir):
    data = generate_array(Type.INT64)
    id = vineyard_client.put(data)
    extracted_data = read_data_from_fuse(
        str(id)[11:28] + ".arrow", vineyard_fuse_mount_dir
    )

    extracted_data = extracted_data.column("a").chunk(0)
    assert_array(data, extracted_data)


def test_fuse_double_array(vineyard_client, vineyard_fuse_mount_dir):
    data = generate_array(Type.DOUBLE)
    id = vineyard_client.put(data)
    extracted_data = read_data_from_fuse(
        str(id)[11:28] + ".arrow", vineyard_fuse_mount_dir
    )

    extracted_data = extracted_data.column("a").chunk(0)
    assert_array(data, extracted_data)


def test_fuse_string_array(vineyard_client, vineyard_fuse_mount_dir):
    data = generate_array(Type.STRING)
    id = vineyard_client.put(data)
    extracted_data = read_data_from_fuse(
        str(id)[11:28] + ".arrow", vineyard_fuse_mount_dir
    )
    extracted_data = extracted_data.column("a").chunk(0)
    assert compare_two_string_array(data, extracted_data), "string array not the same"


def test_fuse_df(vineyard_client, vineyard_fuse_mount_dir):
    data = generate_dataframe()

    id = vineyard_client.put(data)
    extracted_data = read_data_from_fuse(
        str(id)[11:28] + ".arrow", vineyard_fuse_mount_dir
    )
    assert_dataframe(data, extracted_data)
