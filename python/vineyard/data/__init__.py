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

from . import arrow
from . import base
from . import dataframe
from . import default
from . import graph
from . import index
from . import pickle
from . import series
from . import tensor

from vineyard.core.builder import default_builder_context
from vineyard.core.resolver import default_resolver_context
from vineyard.data.default import register_default_types
from vineyard.data.base import register_base_types
from vineyard.data.arrow import register_arrow_types
from vineyard.data.tensor import register_tensor_types
from vineyard.data.index import register_index_types
from vineyard.data.series import register_series_types
from vineyard.data.dataframe import register_dataframe_types
from vineyard.data.graph import register_graph_types


def register_builtin_types(builder_ctx, resolver_ctx):
    register_default_types(builder_ctx, resolver_ctx)
    register_base_types(builder_ctx, resolver_ctx)
    register_arrow_types(builder_ctx, resolver_ctx)
    register_tensor_types(builder_ctx, resolver_ctx)
    register_index_types(builder_ctx, resolver_ctx)
    register_series_types(builder_ctx, resolver_ctx)
    register_dataframe_types(builder_ctx, resolver_ctx)
    register_graph_types(builder_ctx, resolver_ctx)


# Those builtin builders and resolvers will be registered by default, for better
# import-and-play experience.
register_builtin_types(default_builder_context, default_resolver_context)
