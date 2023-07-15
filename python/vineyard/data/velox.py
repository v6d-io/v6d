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

from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa

import lazy_import

velox = lazy_import.lazy_module('pyvelox.pyvelox')


def evaluate(
    expr: Union[str, velox.Expression], table: Union[Dict, pd.DataFrame, pa.Table]
):
    """Evaluate a velox :code:`Expression` on a table in types of
    :code:`pandas.DataFrame`, :code:`pyarrow.Table`, :code:`pyarrow.RecordBatch`,
    or a dictionary of :code:`pyarrow.Array`s and :code:`numpy.ndarray`s.

    e.g.,

        .. code:: python

            import numpy as np
            import pandas as pd

            df = pd.DataFrame({
                'data': np.random.rand(1000),
                'label': np.random.randint(0, 2, 1000),
            })

            # evaluate a velox expression on a pandas dataframe
            evaluate("data > 0.5 and label == 1", df)

    Args:
        expr: The velox expression to evaluate, and be a string or an
            :code:`velox.Expression`.
        table: The table to evaluate on, can be :code:`pandas.DataFrame`,
            :code:`pyarrow.Table`, :code:`pyarrow.RecordBatch`, or a dictionary of
            :code:`pyarrow.Array`s and :code:`numpy.ndarray`s.

    Returns:
        :code:`pyarrow.Array`
    """
    if isinstance(expr, str):
        expr = velox.Expression.from_string(expr)
    names, columns = [], []
    if isinstance(table, Dict):
        for name, value in table.items():
            names.append(name)
            columns.append(value)
    elif isinstance(table, pd.DataFrame):
        for name, value in table.items():
            names.append(name)
            columns.append(pa.array(value))
    elif isinstance(table, (pa.Table, pa.RecordBatch)):
        for name, column in zip(table.schema.names, table.columns):
            names.append(name)
            columns.append(column)
    else:
        raise ValueError(f'unsupported object type {type(object)}')

    velox_columns = []
    for column in columns:
        if isinstance(column, (list, tuple)):
            column = velox.from_list(column)
        elif isinstance(column, np.ndarray):
            column = velox.import_from_arrow(pa.array(column))
        elif isinstance(column, pa.Array):
            column = velox.import_from_arrow(column)
        elif isinstance(column, pa.ChunkedArray):
            if column.num_chunks == 1:
                column = velox.import_from_arrow(column.chunk(0))
            else:
                raise ValueError(
                    "For pyarrow.ChunkedArray, the num_chunks can only be 1, "
                    "but got %s" % column.num_chunks
                )
        assert isinstance(column, velox.BaseVector)
        velox_columns.append(column)

    return velox.export_to_arrow(expr.evaluate(names, velox_columns))
