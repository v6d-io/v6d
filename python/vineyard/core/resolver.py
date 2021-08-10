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
import inspect
import threading

from sortedcontainers import SortedDict

from vineyard._C import IPCClient, RPCClient, ObjectID, Object
from vineyard.core.utils import find_most_precise_match
from vineyard.core.driver import get_current_drivers


class ResolverContext():
    def __init__(self):
        self.__factory = SortedDict()

    def __str__(self) -> str:
        return str(self.__factory)

    def register(self, typename_prefix, resolver):
        self.__factory[typename_prefix] = resolver

    def run(self, obj, **kw):
        typename = obj.meta.typename
        prefix, resolver = find_most_precise_match(typename, self.__factory)
        vineyard_client = kw.pop('__vineyard_client', None)
        if prefix:
            resolver_func_sig = inspect.getfullargspec(resolver)
            if resolver_func_sig.varkw is not None:
                value = resolver(obj, resolver=self, **kw)
            else:
                # don't pass the `**kw`.
                if 'resolver' in resolver_func_sig.args:
                    value = resolver(obj, resolver=self)
                else:
                    value = resolver(obj)
            if value is None:
                # if the obj has been resolved by pybind types, and there's no proper resolver, it
                # shouldn't be an error
                if type(obj) is not Object:
                    return obj

                raise RuntimeError('Unable to construct the object: no proper resolver found: typename is %s' %
                                   obj.meta.typename)

            # associate a reference to the base C++ object
            try:
                setattr(value, '__vineyard_ref', obj)
                setattr(value, '__vineyard_client', vineyard_client)

                # register methods
                get_current_drivers().resolve(value, obj.typename)
            except AttributeError:
                pass

            return value
        return None

    def __call__(self, obj, **kw):
        return self.run(obj, **kw)

    def extend(self, resolvers=None):
        resolver = ResolverContext()
        resolver.__factory = self.__factory.copy()
        if resolvers:
            resolver.__factory.update(resolvers)
        return resolver


default_resolver_context = ResolverContext()

_resolver_context_local = threading.local()
_resolver_context_local.default_resolver = default_resolver_context


def get_current_resolvers():
    ''' Obtain current resolver context.
    '''
    default_resolver = getattr(_resolver_context_local, 'default_resolver', None)
    if not default_resolver:
        default_resolver = default_resolver_context.extend()
    return default_resolver


@contextlib.contextmanager
def resolver_context(resolvers=None, base=None):
    ''' Open a new context for register resolvers, without populting outside global
        environment.

        The :code:`resolver_context` can be useful when users have more than more resolver
        for a certain type, e.g., the :code:`vineyard::Tensor` object can be resolved as
        :code:`numpy.ndarray` or :code:`xgboost::DMatrix`. We could have

        .. code:: python

            def numpy_resolver(obj):
                ...

            default_resolver_context.register('vineyard::Tensor', numpy_resolver)

        and

        .. code:: python

            def xgboost_resolver(obj):
                ...

            default_resolver_context.register('vineyard::Tensor', xgboost_resolver)

        Obviously there's a conflict, and the stackable :code:`resolver_context` could
        help there,

        .. code:: python

            with resolver_context({'vineyard::Tensor', xgboost_resolver}):
                ...

        Assuming the default context resolves :code:`vineyard::Tensor` to :code:`numpy.ndarray`,
        inside the :code:`with resolver_context` the :code:`vineyard::Tensor` will be resolved
        to :code:`xgboost::DMatrix`, and after exiting the context the global environment
        will be restored back as default.

        The :code:`with resolver_context` is nestable as well.

        See Also:
            builder_context
            driver_context
    '''
    current_resolver = get_current_resolvers()
    try:
        resolvers = resolvers or dict()
        base = base or current_resolver
        local_resolver = base.extend(resolvers)
        _resolver_context_local.default_resolver = local_resolver
        yield local_resolver
    finally:
        _resolver_context_local.default_resolver = current_resolver


def get(client, object_id, resolver=None, **kw):
    ''' Get vineyard object as python value.

    .. code:: python

        >>> arr_id = vineyard.ObjectID('00002ec13bc81226')
        >>> arr = client.get(arr_id)
        >>> arr
        array([0, 1, 2, 3, 4, 5, 6, 7])

    Parameters:
        client: IPCClient or RPCClient
            The vineyard client to use.
        object_id: ObjectID
            The object id that will be obtained from vineyard.
        resolver:
            When retrieving vineyard object, an optional *resolver* can be specified.
            If no resolver given, the default resolver context will be used.
        kw:
            User-specific argument that will be passed to the builder.

    Returns:
        A python object that return by the resolver, by resolving an vineyard object.
    '''
    # wrap object_id
    if isinstance(object_id, (int, str)):
        object_id = ObjectID(object_id)

    # run resolver
    obj = client.get_object(object_id)
    meta = obj.meta
    if not meta.islocal and not meta.isglobal:
        raise ValueError("Not a local object: for remote object, you can only get its metadata")
    if resolver is None:
        resolver = get_current_resolvers()
    return resolver(obj, __vineyard_client=client, **kw)


setattr(IPCClient, 'get', get)
setattr(RPCClient, 'get', get)

__all__ = [
    'default_resolver_context',
    'resolver_context',
    'get_current_resolvers',
]
