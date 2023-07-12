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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Tuple

from vineyard.core.builder import BuilderContext
from vineyard.core.resolver import ResolverContext

from .builder import builder_context
from .builder import default_builder_context
from .driver import default_driver_context
from .driver import driver_context
from .resolver import default_resolver_context
from .resolver import resolver_context
from .utils import ReprableString


@contextlib.contextmanager
def context(
    builders: Optional[Dict[type, Callable]] = None,
    resolvers: Optional[Dict[str, Callable]] = None,
    base_builder: Optional[BuilderContext] = None,
    base_resolver: Optional[ResolverContext] = None,
) -> Generator[Tuple[BuilderContext, ResolverContext], Any, Any]:
    """A context manager for temporary builder and resolver registration.

    See Also:
        builder_context
        resolver_context
    """
    with builder_context(builders, base_builder) as builder:
        with resolver_context(resolvers, base_resolver) as resolver:
            yield builder, resolver
