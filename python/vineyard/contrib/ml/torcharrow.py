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

import contextlib

import pandas as pd
import pyarrow as pa

import lazy_import

from vineyard.core import context

ta = lazy_import.lazy_module("torcharrow")


def torcharrow_column_builder(client, value, builder, **kw):
    return builder.run(client, value.to_arrow(), **kw)


def torcharrow_dataframe_builder(client, value, builder, **kw):
    return builder.run(client, value.to_arrow(), **kw)


def torcharrow_column_resolver(obj, resolver, **kw):
    value = resolver.parent_context.run(obj, **kw)
    if isinstance(value, pd.Series):
        return ta.from_pandas(value)
    elif isinstance(value, (pa.Array, pa.ChunkedArray)):
        return ta.from_arrow(value)
    else:
        raise TypeError(f'Unsupported type {type(value)}')


def torcharrow_dataframe_resolver(obj, resolver, **kw):
    value = resolver.parent_context.run(obj, **kw)
    if isinstance(value, pd.DataFrame):
        return ta.from_pandas(value)
    elif isinstance(value, (pa.Table, pa.RecordBatch)):
        return ta.from_arrow(value)
    else:
        raise TypeError(f'Unsupported type {type(value)}')


def register_torcharrow_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(ta.Column, torcharrow_column_builder)
        builder_ctx.register(ta.DataFrame, torcharrow_dataframe_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::NumericArray', torcharrow_column_resolver)
        resolver_ctx.register(
            'vineyard::FixedSizeBinaryArray', torcharrow_column_resolver
        )
        resolver_ctx.register('vineyard::LargeBinaryArray', torcharrow_column_resolver)
        resolver_ctx.register('vineyard::LargeStringArray', torcharrow_column_resolver)
        resolver_ctx.register(
            'vineyard::BaseBinaryArray<arrow::LargeStringArray>',
            torcharrow_column_resolver,
        )
        resolver_ctx.register('vineyard::BooleanArray', torcharrow_column_resolver)
        resolver_ctx.register('vineyard::BooleanArray', torcharrow_column_resolver)
        resolver_ctx.register('vineyard::BooleanArray', torcharrow_column_resolver)
        resolver_ctx.register('vineyard::DataFrame', torcharrow_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', torcharrow_dataframe_resolver)
        resolver_ctx.register('vineyard::Table', torcharrow_dataframe_resolver)


@contextlib.contextmanager
def torcharrow_context():
    with context() as (builder_ctx, resolver_ctx):
        with contextlib.suppress(ImportError):
            register_torcharrow_types(builder_ctx, resolver_ctx)
        yield builder_ctx, resolver_ctx
