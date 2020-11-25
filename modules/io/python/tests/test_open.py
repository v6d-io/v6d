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

''' How to run those test:

    * Step 1: setup a vineyard server:

        vineyardd --socket=/tmp/vineyard.sock

    * Step 2: using pytest to run the following tests:

    .. code:: console

        pytest modules/io/python/tests/test_open.py --vineyard-ipc-socket=/tmp/vineyard.sock \
                                                    --vineyard-endpoint=127.0.0.1:9600 \
                                                    --test-dataset=<directory of gstest>

    If you want to run those HDFS tests, add the following paramters:

    .. code:: console

        pytest modules/io/python/tests/test_open.py --with-hdfs \
                                                    --hdfs-endpoint=hdfs://dev:9000 \
                                                    --hive-endpoint=hive://dev:9000
'''

import filecmp
import pytest
import configparser

import vineyard
import vineyard.io


def test_local_with_header(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp):
    stream = vineyard.io.open('file://%s/p2p-31.e#header_row=true&delimiter= ' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/p2p-31.out' % test_dataset_tmp,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e' % test_dataset, '%s/p2p-31.out' % test_dataset_tmp)


def test_local_without_header(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp):
    stream = vineyard.io.open('file://%s/p2p-31.e#header_row=false&delimiter= ' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/p2p-31.out' % test_dataset_tmp,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e' % test_dataset, '%s/p2p-31.out' % test_dataset_tmp)


def test_local_orc(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp):
    stream = vineyard.io.open('file://%s/p2p-31.e.orc' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/testout.orc' % test_dataset_tmp,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e.orc' % test_dataset, '%s/testout.orc' % test_dataset_tmp)


@pytest.mark.skip_without_hdfs()
def test_hdfs_orc(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp, hdfs_endpoint):
    stream = vineyard.io.open('file://%s/test.orc' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('%s/tmp/testout.orc' % hdfs_endpoint,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    streamout = vineyard.io.open('hdfs://dev:9000/tmp/testout.orc',
                                 vineyard_ipc_socket=vineyard_ipc_socket,
                                 vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/testout1.orc' % test_dataset_tmp,
                     streamout,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/test.orc' % test_dataset, '%s/testout1.orc' % test_dataset_tmp)


@pytest.mark.skip_without_hdfs()
def test_hive(vineyard_ipc_socket, vineyard_endpoint, test_dataset, hive_endpoint):
    stream = vineyard.io.open('%s/user/hive/warehouse/pt' % hive_endpoint,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/testout1.e' % test_dataset,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)


@pytest.mark.skip_without_hdfs()
def test_hdfs_bytes(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp, hdfs_endpoint):
    stream = vineyard.io.open('file://%s/p2p-31.e#header_row=true&delimiter= ' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('%s/tmp/p2p-31.out' % hdfs_endpoint,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    hdfs_stream = vineyard.io.open('hdfs://dev:9000/tmp/p2p-31.out#header_row=true&delimiter= ',
                                   vineyard_ipc_socket=vineyard_ipc_socket,
                                   vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/p2p-31.out' % test_dataset_tmp,
                     hdfs_stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e' % test_dataset, '%s/p2p-31.out' % test_dataset_tmp)


def test_vineyard_dataframe(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp):
    stream = vineyard.io.open('file://%s/p2p-31.e#header_row=false&delimiter= ' % test_dataset,
                              vineyard_ipc_socket=vineyard_ipc_socket,
                              vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('vineyard://p2p-gdf',
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    dfstream = vineyard.io.open('vineyard://p2p-gdf#delimiter= ',
                                vineyard_ipc_socket=vineyard_ipc_socket,
                                vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/p2p-31.out' % test_dataset_tmp,
                     dfstream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e' % test_dataset, '%s/p2p-31.out' % test_dataset_tmp)

@pytest.mark.skip('oss not available at github ci')
def test_oss(vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp, oss_config):
    oss_config_file = oss_config + '/.ossutilconfig'
    oss_config = configparser.ConfigParser()
    oss_config.read(oss_config_file)
    accessKeyID = oss_config['Credentials']['accessKeyID']
    accessKeySecret = oss_config['Credentials']['accessKeySecret']
    oss_endpoint = oss_config['Credentials']['endpoint']
    stream = vineyard.io.open(
        f'oss://{accessKeyID}:{accessKeySecret}@{oss_endpoint}/grape-uk/p2p-31.e#header_row=false&delimiter= ',
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint)
    vineyard.io.open('file://%s/p2p-31.out' % test_dataset_tmp,
                     stream,
                     mode='w',
                     vineyard_ipc_socket=vineyard_ipc_socket,
                     vineyard_endpoint=vineyard_endpoint)
    assert filecmp.cmp('%s/p2p-31.e' % test_dataset, '%s/p2p-31.out' % test_dataset_tmp)
