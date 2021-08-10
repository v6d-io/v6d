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

import contextlib
import copy
import functools
import threading

from sortedcontainers import SortedDict

from .utils import find_most_precise_match


class DriverContext():
    def __init__(self):
        self.__factory = SortedDict(dict)

    def __str__(self) -> str:
        return str(self.__factory)

    def register(self, typename_prefix, meth, func):
        if typename_prefix not in self.__factory:
            self.__factory[typename_prefix] = dict()
        self.__factory[typename_prefix][meth] = func

    def resolve(self, obj, typename):
        prefix, methods = find_most_precise_match(typename, self.__factory)
        if prefix:
            for meth, func in methods.items():
                # if shouldn't failed, since it has already been wrapped in during resolving
                setattr(obj, meth, functools.partial(func, obj))
        return obj

    def __call__(self, obj, typename):
        return self.resolve(obj, typename)

    def extend(self, drivers=None):
        driver = DriverContext()
        driver.__factory.update(((k, copy.copy(v)) for k, v in self.__factory.items()))
        if drivers:
            for ty, methods in drivers.items():
                if ty not in self.__factory:
                    driver.__factory[ty] = dict()
                driver.__factory[ty].update(methods)
        return driver


default_driver_context = DriverContext()

_driver_context_local = threading.local()
_driver_context_local.default_driver = default_driver_context


def get_current_drivers():
    ''' Obtain current driver context.
    '''
    default_driver = getattr(_driver_context_local, 'default_driver', None)
    if not default_driver:
        default_driver = default_driver_context.extend()
    return default_driver


@contextlib.contextmanager
def driver_context(drivers=None, base=None):
    ''' Open a new context for register drivers, without populting outside global
        environment.

        See Also:
            builder_context
            resolver_context
    '''
    current_driver = get_current_drivers()
    try:
        drivers = drivers or dict()
        base = base or current_driver
        local_driver = base.extend(drivers)
        _driver_context_local.default_driver = local_driver
        yield local_driver
    finally:
        _driver_context_local.default_driver = current_driver


def register_builtin_drivers(ctx):
    assert isinstance(ctx, DriverContext)

    # TODO
    # there's no builtin drivers yet.


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


__all__ = [
    'default_driver_context',
    'register_builtin_drivers',
    'driver_context',
    'get_current_drivers',
    'registerize',
]
