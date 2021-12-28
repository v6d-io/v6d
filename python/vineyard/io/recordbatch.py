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

''' This module exposes support for RecordBatchStream.
'''

import json
from typing import Dict

from .._C import ObjectID, ObjectMeta
from .stream import BaseStream


class RecordBatchStream(BaseStream):
    def __init__(self, meta: ObjectMeta, params: Dict = None):
        super().__init__(meta)
        self._params = params

    @property
    def params(self):
        return self._params

    @staticmethod
    def new(client, params: Dict = None) -> "RecordBatchStream":
        meta = ObjectMeta()
        meta['typename'] = 'vineyard::RecordBatchStream'
        if params is None:
            params = dict()
        meta['params'] = params
        meta = client.create_metadata(meta)
        client.create_stream(meta.id)
        return RecordBatchStream(client, meta, params)


def recordbatch_stream_resolver(obj, resolver):
    meta = obj.meta
    if 'params' in meta:
        params = json.loads(meta['params'])
    else:
        params = dict
    return RecordBatchStream(meta)


def register_recordbatch_stream_types(builder_ctx, resolver_ctx):
    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::RecordBatchStream', recordbatch_stream_resolver)
