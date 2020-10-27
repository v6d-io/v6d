#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
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

import numpy as np


def normalize_dtype(dtype):
    ''' Normalize a descriptive C++ type to numpy.dtype.
    '''
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
    if dtype in ['float', 'float32']:
        return np.dtype('float')
    if dtype in [float, 'double', 'float64']:
        return np.dtype('double')
    return dtype


def build_buffer(client, address, size):
    if size == 0:
        return client.create_empty_blob()
    buffer = client.create_blob(size)
    buffer.copy(0, address, size)
    return buffer.seal(client)


def build_numpy_buffer(client, array):
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    address, _ = array.__array_interface__['data']
    return build_buffer(client, address, array.nbytes)
