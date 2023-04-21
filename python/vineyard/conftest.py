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

import os

import pytest

import vineyard


def pytest_addoption(parser):
    parser.addoption(
        "--vineyard-ipc-socket",
        action="store",
        default=None,
        help='Location of vineyard IPC socket',
    )

    parser.addoption(
        "--vineyard-endpoint",
        action="store",
        default=None,
        help='Address of vineyard RPC endpoint',
    )

    parser.addoption(
        "--vineyard-ipc-sockets",
        action="store",
        default=None,
        help='Location of vineyard IPC sockets, separated by ","',
    )

    parser.addoption(
        "--vineyard-endpoints",
        action="store",
        default=None,
        help='Address of vineyard RPC endpoints, separated by ","',
    )

    parser.addoption(
        '--test-dataset',
        action='store',
        default=os.path.expandvars('$VINEYARD_DATA_DIR'),
        help='Location of dataset that will be used for running test cases',
    )

    parser.addoption(
        '--test-dataset-tmp',
        action='store',
        default=os.path.expandvars('$TMPDIR'),
        help='Location of temporary directory that will be used for running test cases',
    )

    parser.addoption(
        '--with-hdfs',
        action='store_true',
        default=False,
        help='Test with HDFS enabled',
    )

    parser.addoption(
        '--hdfs-endpoint',
        action='store',
        default='hdfs://127.0.0.1:9000',
        help='HDFS filesystem that will be used to run vineyard tests',
    )

    parser.addoption(
        '--hive-endpoint',
        action='store',
        default='hive://127.0.0.1:9000',
        help="Hive's hiveserver2 endpoint that will be used to run vineyard tests",
    )

    parser.addoption(
        '--oss-config',
        action='store',
        default=os.path.expandvars('$OSS_CONFIG_DIR'),
        help='Location of oss config to login oss server',
    )

    parser.addoption(
        '--with-migration',
        action='store_true',
        default=False,
        help='Test with object migration enabled',
    )


@pytest.fixture(scope='session')
def vineyard_ipc_sockets(request):
    if request.config.option.vineyard_ipc_sockets:
        return request.config.option.vineyard_ipc_sockets.split(',')
    if request.config.option.vineyard_ipc_socket:
        return [request.config.option.vineyard_ipc_socket]
    return None


@pytest.fixture(scope='session')
def vineyard_endpoints(request):
    if request.config.option.vineyard_endpoints:
        return request.config.option.vineyard_endpoints.split(',')
    if request.config.option.vineyard_endpoint:
        return [request.config.option.vineyard_endpoint]
    return None


@pytest.fixture(scope='session')
def vineyard_ipc_socket(request):
    ipc_socket = request.config.option.vineyard_ipc_socket
    if ipc_socket is None and request.config.option.vineyard_ipc_sockets is not None:
        ipc_socket = request.config.option.vineyard_ipc_sockets.split(',')[0]
    return ipc_socket


@pytest.fixture(scope='session')
def vineyard_endpoint(request):
    rpc_endpoint = request.config.option.vineyard_endpoint
    if rpc_endpoint is None and request.config.option.vineyard_endpoints is not None:
        rpc_endpoint = request.config.option.vineyard_endpoints.split(',')[0]
    return rpc_endpoint


@pytest.fixture(scope='session')
def test_dataset(request):
    return request.config.option.test_dataset


@pytest.fixture(scope='session')
def test_dataset_tmp(request):
    return request.config.option.test_dataset_tmp


@pytest.fixture(scope='session')
def with_hdfs(request):
    return request.config.option.with_hdfs


@pytest.fixture(scope='session')
def hdfs_endpoint(request):
    return request.config.option.hdfs_endpoint


@pytest.fixture(scope='session')
def hive_endpoint(request):
    return request.config.option.hive_endpoint


@pytest.fixture(scope='session')
def oss_config(request):
    return request.config.option.oss_config


@pytest.fixture(scope='session')
def with_migration(request):
    return request.config.option.with_migration


@pytest.fixture(scope='session')
def vineyard_client(request):
    ipc_socket = request.config.option.vineyard_ipc_socket
    if ipc_socket is not None:
        return vineyard.connect(socket=ipc_socket)
    else:
        return None


@pytest.fixture(scope='session')
def vineyard_rpc_client(request):
    rpc_endpoint = request.config.option.vineyard_endpoint
    if rpc_endpoint is not None:
        return vineyard.connect(endpoint=tuple(rpc_endpoint.split(':')))
    else:
        return None


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'skip_without_hdfs(): skip HDFS test if HDFS service is not available',
    )

    config.addinivalue_line(
        'markers',
        'skip_without_migration(): skip migration tests if object migration is '
        'not available',
    )


def pytest_runtest_setup(item):
    markers = [mark for mark in item.iter_markers(name='skip_without_hdfs')]
    if markers:
        if not item.config.option.with_hdfs:
            pytest.skip('Skip since HDFS service is not available')

    markers = [mark for mark in item.iter_markers(name='skip_without_migration')]
    if markers:
        if not item.config.option.with_migration:
            pytest.skip('Skip since object migration is not available')


pytest_plugins = []


# suppress "no test selected error"
def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0


def pytest_collection_modifyitems(items):
    for item in items:
        timeout_marker = None
        if hasattr(item, "get_closest_marker"):
            timeout_marker = item.get_closest_marker("timeout")
        elif hasattr(item, "get_marker"):
            timeout_marker = item.get_marker("timeout")
        if timeout_marker is None:
            item.add_marker(pytest.mark.timeout(600))
