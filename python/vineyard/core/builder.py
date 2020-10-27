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

import inspect

from vineyard._C import IPCClient, RPCClient


class BuilderContext():
    def __init__(self):
        self.__factory = dict()

    def register(self, type_id, builder):
        ''' Register a Python type to the builder context.

            Parameters
            ----------
            type_id: Python type
                Like `int`, or `numpy.ndarray`

            builder: callable, e.g., a method, callable object
                A builder translates a python object to vineyard, it accepts a Python
                value as parameter, and returns an vineyard object as result.
        '''
        self.__factory[type_id] = builder

    def run(self, client, value, **kw):
        ''' Follows the order of MRO to find the proper builder for given python value.

            Here "Follows the order of MRO" implies:

            - If the type of python value has been found in the context, the registered
              builder will be used.

            - If not, it follows the MRO chain from down to top to find a registered
              Python type and used the associated builder.

            - When the traversal reaches the :code:`object` type, since there's a default
              builder that serailization the python value, the parameter will be serialized
              and be put into a blob.
        '''
        for ty in type(value).__mro__:
            if ty in self.__factory:
                builder_func_sig = inspect.getfullargspec(self.__factory[ty])
                if 'builder' in builder_func_sig.args or builder_func_sig.varkw is not None:
                    kw['builder'] = self
                return self.__factory[ty](client, value, **kw)
        raise RuntimeError('Unknown type to build as vineyard object')


default_builder_context = BuilderContext()


def put(client, value, builder=None, **kw):
    ''' Put python value to vineyard.

    .. code:: python

        >>> arr = np.arange(8)
        >>> id = client.put(arr)
        >>> id
        00002ec13bc81226

    Parameters:
        client: IPCClient
            The vineyard client to use.
        value:
            The python value that will be put to vineyard. Supported python value types are
            decided by modules that registered to vineyard. By default, python value can be
            put to vineyard after serialized as a bytes buffer using pickle, or pyarrow, when
            apache-arrow is installed.
        builder:
            When putting python value to vineyard, an optional *builder* can be specified to
            tell vineyard how to construct the corresponding vineyard :class:`Object`. If not
            specified, the default builder context will be used to select a proper builder.
        kw:
            User-specific argument that will be passed to the builder.

    Returns:
        ObjectID: The result object id will be returned.
    '''
    if builder is not None:
        return builder(client, value, **kw)
    return default_builder_context.run(client, value, **kw)


setattr(IPCClient, 'put', put)
setattr(RPCClient, 'put', put)

__all__ = ['default_builder_context']
