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
import inspect
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional

from sortedcontainers import SortedDict

from vineyard._C import IPCClient
from vineyard._C import Object
from vineyard._C import ObjectID
from vineyard._C import RPCClient
from vineyard.core.utils import find_most_precise_match


class ResolverContext:
    def __init__(self, parent_context: Optional["ResolverContext"] = None):
        self._factory = SortedDict()
        if parent_context is not None:
            self._parent_context = parent_context
        else:
            self._parent_context = self

    def __str__(self) -> str:
        return str(self._factory)

    @property
    def parent_context(self) -> "ResolverContext":
        return self._parent_context

    def register(self, typename_prefix: str, resolver: Callable):
        self._factory[typename_prefix] = resolver

    def run(self, obj: Any, **kw):
        typename = obj.meta.typename
        prefix, resolver = find_most_precise_match(typename, self._factory)
        vineyard_client = kw.pop('__vineyard_client', None)
        if prefix:
            resolver_func_sig = inspect.getfullargspec(resolver)
            if resolver_func_sig.varkw is not None:
                value = resolver(obj, resolver=self, **kw)
            else:
                try:
                    # don't pass the `**kw`.
                    if 'resolver' in resolver_func_sig.args:
                        value = resolver(obj, resolver=self)
                    else:
                        value = resolver(obj)
                except Exception as e:
                    raise RuntimeError(  # pylint: disable=raise-missing-from
                        'Unable to construct the object using resolver: '
                        'typename is %s, resolver is %s' % (obj.meta.typename, resolver)
                    ) from e
            if value is None:
                # if the obj has been resolved by pybind types, and there's no proper
                # resolver, it shouldn't be an error
                if type(obj) is not Object:  # pylint: disable=unidiomatic-typecheck
                    return obj

                # we might `client.put(None)`
                return None

            # associate a reference to the base C++ object
            try:
                setattr(value, '__vineyard_ref', obj)
                setattr(value, '__vineyard_client', vineyard_client)

                # register methods
                from vineyard.core.driver import get_current_drivers

                get_current_drivers().resolve(value, obj.typename)
            except AttributeError:
                pass

            return value
        # keep it as it is
        return obj

    def __call__(self, obj, **kw):
        return self.run(obj, **kw)

    def extend(self, resolvers=None):
        resolver = ResolverContext(self)
        resolver._factory = (  # pylint: disable=unused-private-member
            self._factory.copy()
        )
        if resolvers:
            resolver._factory.update(resolvers)
        return resolver


default_resolver_context = ResolverContext()

_resolver_context_local = threading.local()
_resolver_context_local.default_resolver = default_resolver_context


def get_current_resolvers() -> ResolverContext:
    '''Obtain current resolver context.'''
    default_resolver = getattr(_resolver_context_local, 'default_resolver', None)
    if not default_resolver:
        default_resolver = default_resolver_context.extend()
    return default_resolver


@contextlib.contextmanager
def resolver_context(
    resolvers: Optional[Dict[str, Callable]] = None,
    base: Optional[ResolverContext] = None,
) -> Generator[ResolverContext, Any, Any]:
    """Open a new context for register resolvers, without populating outside
    the global environment.

    The :code:`resolver_context` can be useful when users have more than
    more resolver for a certain type, e.g., the :code:`vineyard::Tensor`
    object can be resolved as :code:`numpy.ndarray` or :code:`xgboost::DMatrix`.

    We could have

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

    Assuming the default context resolves :code:`vineyard::Tensor` to
    :code:`numpy.ndarray`, inside the :code:`with resolver_context` the
    :code:`vineyard::Tensor` will be resolved to :code:`xgboost::DMatrix`,
    and after exiting the context the global environment will be restored
    back as default.

    The :code:`with resolver_context` is nestable as well.

    See Also:
        builder_context
        driver_context
    """
    current_resolver = get_current_resolvers()
    try:
        resolvers = resolvers or dict()
        base = base or current_resolver
        local_resolver = base.extend(resolvers)
        _resolver_context_local.default_resolver = local_resolver
        yield local_resolver
    finally:
        _resolver_context_local.default_resolver = current_resolver


def get(
    client,
    object_id: Optional[ObjectID] = None,
    name: Optional[str] = None,
    resolver: Optional[ResolverContext] = None,
    fetch: bool = False,
    **kw
):
    """Get vineyard object as python value.

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
        name: ObjectID
            The object name that will be obtained from vineyard, ignored if
            ``object_id`` is not None.
        resolver:
            When retrieving vineyard object, an optional *resolver* can be specified.
            If no resolver given, the default resolver context will be used.
        fetch:
            Whether to trigger a migration when the target object is located on
            remote instances.
        kw:
            User-specific argument that will be passed to the builder.

    Returns:
        A python object that return by the resolver, by resolving an vineyard object.
    """
    # wrap object_id
    if object_id is not None:
        if isinstance(object_id, (int, str)):
            object_id = ObjectID(object_id)
    elif name is not None:
        object_id = client.get_name(name)

    # run resolver
    obj = client.get_object(object_id, fetch=fetch)
    meta = obj.meta
    if not meta.islocal and not meta.isglobal:
        raise ValueError(
            "Not a local object: for remote object, you can only get its metadata"
        )
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
