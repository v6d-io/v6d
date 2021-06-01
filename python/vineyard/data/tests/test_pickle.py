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

import pandas as pd
import pytest
import numpy as np

from vineyard.data.pickle import PickledReader, PickledWriter

b1m = 1 * 1024 * 1024
b16m = 16 * 1024 * 1024
b64m = 64 * 1024 * 1024
b128m = 128 * 1024 * 1024

values = [
    (b1m, 1),
    (b1m, (1, 2, 3)),
    (b1m, [1, 2, 3, 4]),
    (b1m, "dsdfsdf"),
    (b1m, (1, "sdfsdfs")),
    (b1m, [1] * 100000000),
    (b1m, np.arange(1024 * 1024 * 400)),
    (b16m, np.zeros((1024, 1024, 48))),
    (b64m, np.zeros((1024, 1024, 512))),
    (b1m, pd.DataFrame({
        'a': np.ones(1024),
        'b': np.zeros(1024),
    })),
    (b16m, pd.DataFrame({
        'a': np.ones(1024 * 1024),
        'b': np.zeros(1024 * 1024),
    })),
    (b64m, pd.DataFrame({
        'a': np.ones(1024 * 1024 * 4),
        'b': np.zeros(1024 * 1024 * 4),
    })),
    (b128m, pd.DataFrame({
        'a': np.ones(1024 * 1024 * 16),
        'b': np.zeros(1024 * 1024 * 16),
    })),
]


@pytest.mark.parametrize("block_size, value", values)
def test_bytes_io_roundtrip(block_size, value):
    reader = PickledReader(value)
    bs, nlen = [], 0
    while True:
        block = reader.read(block_size)
        if block:
            bs.append(block)
            nlen += len(block)
        else:
            break
    assert nlen == reader.store_size

    writer = PickledWriter(reader.store_size)
    for block in bs:
        writer.write(block)
    assert writer.value is None
    writer.close()
    assert writer.value is not None
    target = writer.value

    # compare values
    if isinstance(value, np.ndarray):
        # FIXME why `assert_array_equal` are so slow ...
        #
        # np.testing.assert_array_equal(target, value)
        #
        assert (target == value).all()
    elif isinstance(value, pd.DataFrame):
        pd.testing.assert_frame_equal(target, value)
    elif isinstance(value, pd.Index):
        pd.testing.assert_index_equal(target, value)
    elif isinstance(value, pd.Series):
        pd.testing.assert_series_equal(target, value)
    else:
        assert target == value
