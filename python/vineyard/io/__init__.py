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

from vineyard.core.builder import default_builder_context
from vineyard.core.resolver import default_resolver_context

from . import byte
from . import dataframe
from . import parallel
from . import recordbatch
from . import stream
from .byte import register_byte_stream_types
from .dataframe import register_dataframe_stream_types
from .parallel import register_parallel_stream_types
from .recordbatch import register_recordbatch_stream_types
from .serialization import deserialize
from .serialization import serialize
from .stream import open
from .stream import read
from .stream import register_stream_collection_types
from .stream import write


def register_builtin_stream_types(builder_ctx, resolver_ctx):
    register_byte_stream_types(builder_ctx, resolver_ctx)
    register_dataframe_stream_types(builder_ctx, resolver_ctx)
    register_parallel_stream_types(builder_ctx, resolver_ctx)
    register_recordbatch_stream_types(builder_ctx, resolver_ctx)
    register_stream_collection_types(builder_ctx, resolver_ctx)


# Those builtin builders and resolvers will be registered by default, for better
# import-and-play experience.
register_builtin_stream_types(default_builder_context, default_resolver_context)
