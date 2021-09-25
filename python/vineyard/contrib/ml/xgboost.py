#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
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
import xgboost as xgb

from vineyard.core.resolver import resolver_context, default_resolver_context


def xgb_builder(client, value, builder, **kw):
    # TODO: build DMatrix to vineyard objects
    pass


def xgb_tensor_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        array = resolver(obj, **kw)
    return xgb.DMatrix(array)


def xgb_dataframe_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        df = resolver(obj, **kw)
    if 'label' in kw:
        label = df.pop(kw['label'])
        # data column can only be specified if label column is specified
        if 'data' in kw:
            df = np.stack(df[kw['data']].values)
        return xgb.DMatrix(df, label)
    return xgb.DMatrix(df, feature_names=df.columns)


def xgb_recordBatch_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        rb = resolver(obj, **kw)
    # FIXME to_pandas is not zero_copy guaranteed
    df = rb.to_pandas()
    if 'label' in kw:
        label = df.pop(kw['label'])
        return xgb.DMatrix(df, label, feature_names=df.columns)
    return xgb.DMatrix(df, feature_names=df.columns)


def xgb_table_resolver(obj, **kw):
    with resolver_context(base=default_resolver_context) as resolver:
        tb = resolver(obj, **kw)

    # FIXME to_pandas is not zero_copy guaranteed
    df = tb.to_pandas()
    if 'label' in kw:
        label = df.pop(kw['label'])
        return xgb.DMatrix(df, label, feature_names=df.columns)
    return xgb.DMatrix(df, feature_names=df.columns)


def register_xgb_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(xgb.DMatrix, xgb_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Tensor', xgb_tensor_resolver)
        resolver_ctx.register('vineyard::DataFrame', xgb_dataframe_resolver)
        resolver_ctx.register('vineyard::RecordBatch', xgb_recordBatch_resolver)
        resolver_ctx.register('vineyard::Table', xgb_table_resolver)
