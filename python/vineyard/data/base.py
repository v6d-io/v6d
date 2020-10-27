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
import re

import vineyard
from vineyard._C import Object, ObjectMeta
from .utils import normalize_dtype


class ObjectSet:
    def __init__(self, object_or_meta):
        if isinstance(object_or_meta, Object):
            self.meta = object_or_meta.meta
        else:
            self.meta = object_or_meta

    @property
    def num_of_instances(self):
        return int(self.meta['num_of_instances'])

    @property
    def num_of_objects(self):
        return int(self.meta['num_of_objects'])

    def __getitem__(self, idx):
        ''' Return the member of ith element.
        '''
        return self.meta.get_member('object_%d' % idx)

    def get_member(self, idx):
        ''' Return the member of ith element.
        '''
        return self.meta.get_member('object_%d' % idx)

    def get_member_meta(self, idx):
        ''' Return the member of ith element.
        '''
        return self.meta['object_%d' % idx]


def int_builder(client, value):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Scalar<int>'
    meta['value_'] = value
    meta['type_'] = getattr(type(value), '__name__')
    meta['nbytes'] = 0
    return client.create_metadata(meta)


def double_builder(client, value):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Scalar<double>'
    meta['value_'] = value
    meta['type_'] = getattr(type(value), '__name__')
    meta['nbytes'] = 0
    return client.create_metadata(meta)


def string_builder(client, value):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::Scalar<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>'
    meta['value_'] = value
    meta['type_'] = getattr(type(value), '__name__')
    meta['nbytes'] = 0
    return client.create_metadata(meta)


def tuple_builder(client, value, builder):
    if len(value) == 2:
        # use pair
        meta = ObjectMeta()
        meta['typename'] = 'vineyard::Pair'
        meta.add_member('first_', builder.run(client, value[0]))
        meta.add_member('second_', builder.run(client, value[1]))
        return client.create_metadata(meta)
    else:
        meta = ObjectMeta()
        meta['typename'] = 'vineyard::Tuple'
        meta['size_'] = 3
        for i, item in enumerate(value):
            meta.add_member('__elements_-%d' % i, builder.run(client, item))
        meta['__elements_-size'] = 3
        return client.create_metadata(meta)


def scalar_resolver(obj):
    meta = obj.meta
    typename = obj.typename
    if typename == 'vineyard::Scalar<std::basic_string<char,std::char_traits<char>,std::allocator<char>>>':
        return meta['value_']
    if typename == 'vineyard::Scalar<int>':
        return int(meta['value_'])
    if typename == 'vineyard::Scalar<float>' or typename == 'vineyard::Scalar<double>':
        return float(meta['value_'])
    return None


def pair_resolver(obj, resolver):
    fst = obj.member('first_')
    snd = obj.member('second_')
    return (resolver.run(fst), resolver.run(snd))


def tuple_resolver(obj, resolver):
    meta = obj.meta
    elements = []
    for i in range(int(meta['__elements_-size'])):
        elements.append(resolver.run(obj.member('__elements_-%d' % i)))
    return tuple(elements)


def array_resolver(obj):
    typename = obj.typename
    value_type = normalize_dtype(re.match(r'vineyard::Array<([^>]+)>', typename).groups()[0])
    return np.frombuffer(memoryview(obj.member("buffer_")), dtype=value_type)


def object_set_resolver(obj):
    return ObjectSet(obj)


def register_base_types(builder_ctx=None, resolver_ctx=None):
    if builder_ctx is not None:
        builder_ctx.register(int, int_builder)
        builder_ctx.register(float, double_builder)
        builder_ctx.register(str, string_builder)
        builder_ctx.register(tuple, tuple_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::Scalar', scalar_resolver)
        resolver_ctx.register('vineyard::Pair', pair_resolver)
        resolver_ctx.register('vineyard::Tuple', tuple_resolver)
        resolver_ctx.register('vineyard::Array', array_resolver)
        resolver_ctx.register('vineyard::ObjectSet', object_set_resolver)
