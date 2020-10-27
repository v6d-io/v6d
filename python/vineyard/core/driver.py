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

from collections import defaultdict
import functools

from sortedcontainers import SortedDict

from .utils import find_most_precise_match


class DriverContext():
    def __init__(self):
        self.__factory = defaultdict(SortedDict)

    def register(self, typename_prefix, meth, func):
        self.__factory[meth][typename_prefix] = func

    def resolve(self, obj, typename):
        for meth_name, methods in self.__factory.items():
            prefix, method = find_most_precise_match(typename, methods)
            if prefix is not None:
                meth = functools.partial(method, obj)
                # if shouldn't failed, since it has already been wrapped in during resolving
                setattr(obj, meth_name, meth)
        return obj


default_driver_context = DriverContext()


def repartition(g):
    raise NotImplementedError('No repartition method implementation yet')


def register_builtin_drivers(ctx):
    assert isinstance(ctx, DriverContext)
    ctx.register('vineyard::Graph', 'repartition', repartition)


def registerize(func):
    ''' Registerize a method, add a `__factory` attribute and a `register`
        interface to a method.

        multiple-level register is automatically supported, users can

        >>> open.register(local_io_adaptor)
        >>> open.register(oss_io_adaptor)

        OR

        >>> open.register('file', local_io_adaptor)
        >>> open.register('odps', odps_io_adaptor)

        OR

        >>> open.register('file', 'csv', local_csv_reader)
        >>> open.register('file', 'tsv', local_tsv_reader)
    '''
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    setattr(wrap, '__factory', None)

    def register(*args):
        if len(args) == 1:
            if wrap.__factory is None:
                wrap.__factory = []
            if not isinstance(wrap.__factory, list):
                raise RuntimeError('Invalid arguments: inconsistent with existing registerations')
            wrap.__factory.append(args[0])
        else:
            if wrap.__factory is None:
                wrap.__factory = {}
            if not isinstance(wrap.__factory, dict):
                raise RuntimeError('Invalid arguments: inconsistent with existing registerations')
            root = wrap.__factory
            for arg in args[:-2]:
                if arg not in root:
                    root[arg] = dict()
                root = root[arg]
            if args[-2] not in root:
                root[args[-2]] = list()
            root[args[-2]].append(args[-1])

    setattr(wrap, 'register', register)

    return wrap


__all__ = ['default_driver_context', 'register_builtin_drivers', 'registerize']
