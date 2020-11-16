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

from .version import __version__

import logging
import traceback

# Note [Import pyarrow before _C]
#
# Vineyard's C++ library requires arrow, aka. libarrow.so. The arrow library
# usually be built as a shared library and dynamic-linked into libvineyard_client.so
# and then _C.so.
#
# However pyarrow has its own bundled (static-linked) arrow library, thus
# if we import vineyard's C extension first then import pyarrow there will
# be a DLL version conflict.
#
# Thus we import pyarrow before import vineyard's C extension library.
#
# Note that this only happens on development environment where an apache-arrow's
# shared library has already been installed and we build vineyard on such
# environment. The vineyard's release wheels doesn't suffers such issue since
# we use a static apache-arrow library during building wheels, both for the
# manylinux1 platform and MacOS.


def _init_global_context():
    import os as _dl_flags
    import sys

    if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_LAZY'):
        try:
            # next try if DLFCN exists
            import DLFCN as _dl_flags
        except ImportError:
            _dl_flags = None

    if _dl_flags is not None:
        old_flags = sys.getdlopenflags()

        # See Note [Import pyarrow before _C]
        sys.setdlopenflags(_dl_flags.RTLD_LOCAL | _dl_flags.RTLD_NOW)
        import pyarrow
        del pyarrow

        # import the extension module
        sys.setdlopenflags(_dl_flags.RTLD_LOCAL | _dl_flags.RTLD_NOW | _dl_flags.RTLD_DEEPBIND)
        from . import _C

        # restore
        sys.setdlopenflags(old_flags)


_init_global_context()
del _init_global_context


from ._C import connect, IPCClient, RPCClient, Object, ObjectBuilder, ObjectID, ObjectMeta, \
    InstanceStatus, Blob, BlobBuilder
from ._C import ArrowErrorException, \
    AssertionFailedException, \
    ConnectionErrorException, \
    ConnectionFailedException, \
    EndOfFileException, \
    EtcdErrorException, \
    IOErrorException, \
    InvalidException, \
    InvalidStreamStateException, \
    KeyErrorException, \
    MetaTreeInvalidException, \
    MetaTreeLinkInvalidException, \
    MetaTreeNameInvalidException, \
    MetaTreeNameNotExistsException, \
    MetaTreeSubtreeNotExistsException, \
    MetaTreeTypeInvalidException, \
    MetaTreeTypeNotExistsException, \
    NotEnoughMemoryException, \
    NotImplementedException, \
    ObjectExistsException, \
    ObjectNotExistsException, \
    ObjectNotSealedException, \
    ObjectSealedException, \
    StreamDrainedException, \
    StreamFailedException, \
    TypeErrorException, \
    UnknownErrorException, \
    UserInputErrorException, \
    VineyardServerNotReadyException

from . import _vineyard_docs
del _vineyard_docs

from .core import default_builder_context, default_resolver_context, default_driver_context
from .data import register_builtin_types
from .data.base import ObjectSet
from .data.graph import Graph

logger = logging.getLogger('vineyard')


def _init_vineyard_modules():
    ''' Resolve registered vineyard modules in the following order:

        * /etc/vineyard/config.py
        * {sys.prefix}/etc/vineyard/config.py
        * /usr/share/vineyard/01-xxx.py
        * /usr/local/share/vineyard/01-xxx.py
        * {sys.prefix}/share/vineyard/02-xxxx.py
        * $HOME/.vineyard/03-xxxxx.py
    '''

    import glob
    import importlib.util
    import os
    import sys

    def _import_module_from_file(filepath):
        filepath = os.path.expanduser(os.path.expandvars(filepath))
        if os.path.exists(filepath):
            try:
                spec = importlib.util.spec_from_file_location("vineyard._contrib", filepath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:  # pylint: disable=broad-except
                logger.debug("Failed to load %s: %s\n%s", filepath, e, traceback.format_exc())

    _import_module_from_file('/etc/vineyard/config.py')
    _import_module_from_file(os.path.join(sys.prefix, '/etc/vineyard/config.py'))
    for filepath in glob.glob('/usr/share/vineyard/*-*.py'):
        _import_module_from_file(filepath)
    for filepath in glob.glob('/usr/local/share/vineyard/*-*.py'):
        _import_module_from_file(filepath)
    for filepath in glob.glob(os.path.join(sys.prefix, '/share/vineyard/*-*.py')):
        _import_module_from_file(filepath)
    for filepath in glob.glob(os.path.expanduser('$HOME/.vineyard/*-*.py')):
        _import_module_from_file(filepath)


try:
    _init_vineyard_modules()
except:
    pass
del _init_vineyard_modules
