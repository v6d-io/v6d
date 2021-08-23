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

import pytest

from vineyard.conftest import vineyard_ipc_sockets, vineyard_endpoints


def pytest_addoption(parser):
    parser.addoption(
        '--with-migration',
        action='store_true',
        default=False,
        help='Test with object migration enabled',
    )


@pytest.fixture(scope='session')
def with_migration(request):
    return request.config.option.with_migration


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_without_migration(): skip migration tests if object migration is not available",
    )


def pytest_runtest_setup(item):
    markers = [mark for mark in item.iter_markers(name='skip_without_migration')]
    if markers:
        if not item.config.option.with_migration:
            pytest.skip('Skip since object migration is not available')


pytest_plugins = []
