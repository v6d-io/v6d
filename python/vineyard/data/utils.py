#! /usr/bin/env python
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

import json
import pickle

import numpy as np
import pyarrow as pa

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle  # pylint: disable=import-error


def normalize_dtype(
    dtype, dtype_meta=None
):  # pylint: disable=too-many-return-statements
    '''Normalize a descriptive C++ type to numpy.dtype.'''
    if isinstance(dtype, np.dtype):
        return dtype
    if dtype in ['i32', 'int', 'int32', 'int32_t']:
        return np.dtype('int32')
    if dtype in ['u32', 'uint', 'uint_t', 'uint32', 'uint32_t']:
        return np.dtype('uint32')
    if dtype in [int, 'i64', 'int64', 'long long', 'int64_t']:
        return np.dtype('int64')
    if dtype in ['u64', 'uint64', 'uint64_t']:
        return np.dtype('uint64')
    if dtype in ['float32']:
        return np.dtype('float32')
    if dtype in ['float']:
        return np.dtype('float')
    if dtype in ['double', 'float64']:
        return np.dtype('double')
    if dtype.startswith('str'):
        return np.dtype(dtype_meta)
    return dtype


def normalize_cpptype(dtype):  # pylint: disable=too-many-return-statements
    if dtype.name == 'int32':
        return 'int'
    if dtype.name == 'uint32':
        return 'uint32'
    if dtype.name == 'int64':
        return 'int64'
    if dtype.name == 'uint64':
        return 'uint64'
    if dtype.name == 'float32':
        return 'float'
    if dtype.name == 'float64':
        return 'double'
    return dtype.name


def normalize_arrow_dtype(  # noqa: C901, pylint: disable=too-many-return-statements
    dtype,
):
    if dtype in ['bool']:
        return pa.bool_()
    if dtype in ['int8_t', 'int8', 'byte']:
        return pa.int8()
    if dtype in ['uint8_t', 'uint8', 'char']:
        return pa.uint8()
    if dtype in ['int16_t', 'int16', 'short']:
        return pa.int16()
    if dtype in ['uint16_t', 'uint16']:
        return pa.uint16()
    if dtype in ['int32_t', 'int32', 'int']:
        return pa.int32()
    if dtype in ['uint32_t', 'uint32']:
        return pa.uint32()
    if dtype in ['int64_t', 'int64', 'long']:
        return pa.int64()
    if dtype in ['uint64_t', 'uint64']:
        return pa.uint64()
    if dtype in ['half']:
        return pa.float16()
    if dtype in ['float', 'float32']:
        return pa.float32()
    if dtype in ['double', 'float64']:
        return pa.float64()
    if dtype in [
        'string',
        'std::string',
        'std::__1::string',
        'std::__cxx11::string',
        'str',
    ]:
        return pa.large_string()
    if dtype in ['large_list<item: int32>']:
        return pa.large_list(pa.int32())
    if dtype in ['large_list<item: uint32>']:
        return pa.large_list(pa.uint32())
    if dtype in ['large_list<item: int64>']:
        return pa.large_list(pa.int64())
    if dtype in ['large_list<item: uint64>']:
        return pa.large_list(pa.uint64())
    if dtype in ['large_list<item: float>']:
        return pa.large_list(pa.float())
    if dtype in ['large_list<item: double>']:
        return pa.large_list(pa.double())
    if dtype in ['null', 'NULL', 'None', None]:
        return pa.null()
    raise ValueError('Unsupported data type: %s' % dtype)


def build_buffer(client, address, size):
    if size == 0:
        return client.create_empty_blob()
    buffer = client.create_blob(size)
    buffer.copy(0, address, size)
    return buffer.seal(client)


def build_numpy_buffer(client, array):
    if array.dtype.name != 'object':
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
        address, _ = array.__array_interface__['data']
        return build_buffer(client, address, array.nbytes)
    else:
        payload = pickle.dumps(array, protocol=5)
        buffer = client.create_blob(len(payload))
        buffer.copy(0, payload)
        return buffer.seal(client)


def default_json_encoder(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError


def to_json(value):
    return json.dumps(value, default=default_json_encoder)


def from_json(string):
    return json.loads(string)


def expand_slice(indexer):
    if isinstance(indexer, slice):
        return range(indexer.start, indexer.stop, indexer.step)
    else:
        return indexer


def str_to_bool(s):
    if s is None:
        return False
    if not isinstance(s, str):
        s = str(s)
    return s.lower() in [
        'true',
        '1',
        't',
        'y',
        'yes',
        'yeah',
        'yup',
        'certainly',
        'uh-huh',
    ]


__all__ = [
    'normalize_dtype',
    'normalize_cpptype',
    'normalize_arrow_dtype',
    'build_buffer',
    'build_numpy_buffer',
    'to_json',
    'from_json',
    'expand_slice',
    'str_to_bool',
]
