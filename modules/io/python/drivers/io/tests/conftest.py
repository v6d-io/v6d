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

import logging
import os

import pytest

import vineyard

logging.basicConfig(level=logging.NOTSET)


def pytest_addoption(parser):
    parser.addoption(
        '--vineyard-ipc-socket',
        action='store',
        default='/tmp/vineyard.sock',
        help='Location of vineyard IPC socket',
    )

    parser.addoption(
        "--vineyard-ipc-sockets",
        action="store",
        default='/tmp/vineyard.sock',
        help='Location of vineyard IPC sockets, seperated by ","',
    )

    parser.addoption(
        '--vineyard-endpoint',
        action='store',
        default='127.0.0.1:9600',
        help='Address of vineyard RPC endpoint',
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
def vineyard_ipc_socket(request):
    return request.config.option.vineyard_ipc_socket


@pytest.fixture(scope='session')
def vineyard_ipc_sockets(request):
    return request.config.option.vineyard_ipc_sockets.split(',')


@pytest.fixture(scope='session')
def vineyard_endpoint(request):
    return request.config.option.vineyard_endpoint


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
    rpc_endpoint = request.config.option.vineyard_rpc_endpoint
    if rpc_endpoint is not None:
        return request.connect(rpc_endpoint)
    else:
        return vineyard.connect(ipc_socket)


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
