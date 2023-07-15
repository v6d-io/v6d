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

import numpy as np
import pandas as pd
import pyarrow as pa

import pytest

try:
    import pyvelox.pyvelox as velox
except ImportError:
    velox = None
else:
    from vineyard.data.velox import evaluate


@pytest.mark.skipif(velox is None, reason='pyvelox is not installed')
def test_velox_dataframe():
    df = pd.DataFrame(
        {
            'data': np.random.rand(1000),
            'label': np.random.randint(0, 2, 1000),
        }
    )
    result = evaluate("data > 0.5 and label == 1", df)

    assert result == pa.array((df['data'] > 0.5) & (df['label'] == 1))


@pytest.mark.skipif(velox is None, reason='pyvelox is not installed')
def test_velox_table():
    df = pd.DataFrame(
        {
            'data': np.random.rand(1000),
            'label': np.random.randint(0, 2, 1000),
        }
    )
    result = evaluate("data > 0.5 and label == 1", pa.Table.from_pandas(df))

    assert result == pa.array((df['data'] > 0.5) & (df['label'] == 1))


@pytest.mark.skipif(velox is None, reason='pyvelox is not installed')
def test_velox_dict():
    df = pd.DataFrame(
        {
            'data': np.random.rand(1000),
            'label': np.random.randint(0, 2, 1000),
        }
    )
    result = evaluate(
        "data > 0.5 and label == 1",
        {
            'data': df['data'].values,
            'label': df['label'].values,
        },
    )

    assert result == pa.array((df['data'] > 0.5) & (df['label'] == 1))
