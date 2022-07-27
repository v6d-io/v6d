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

""" How to run those test:

    * Step 1: setup a vineyard server:

        vineyardd --socket=/tmp/vineyard.sock

    * Step 2: using pytest to run the following tests:

    .. code:: console

        pytest modules/io/python/drivers/io/tests/test_open.py \
                --vineyard-ipc-socket=/tmp/vineyard.sock \
                --vineyard-endpoint=127.0.0.1:9600 \
                --test-dataset=<directory of gstest>

    If you want to run those HDFS tests, add the following paramters:

    .. code:: console

        pytest modules/io/python/drivers/io/tests/test_open.py \
                --with-hdfs \
                --hdfs-endpoint=hdfs://dev:9000 \
                --hive-endpoint=hive://dev:9000
"""

import filecmp
import glob
import os
from urllib.parse import urlparse

import pytest

import vineyard
import vineyard.io


def test_local_with_header(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={"header_row": True, "delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


def test_local_without_header(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={"header_row": False, "delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


@pytest.mark.skip()
def test_local_orc(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    stream = vineyard.io.open(
        "file://%s/p2p-31.e.orc" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    vineyard.io.open(
        "file://%s/testout.orc" % test_dataset_tmp,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e.orc" % test_dataset, "%s/testout.orc_0" % test_dataset_tmp
    )


@pytest.mark.skip_without_hdfs()
def test_hdfs_orc(
    vineyard_ipc_socket,
    vineyard_endpoint,
    test_dataset,
    test_dataset_tmp,
    hdfs_endpoint,
):
    stream = vineyard.io.open(
        "file://%s/p2p-31.e.orc" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    res = urlparse(hdfs_endpoint)
    host, port = res.netloc.split(":")
    port = int(port)
    vineyard.io.open(
        "%s:///tmp/testout.orc" % res.scheme,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
    )
    streamout = vineyard.io.open(
        "%s:///tmp/testout.orc_0" % res.scheme,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
    )
    vineyard.io.open(
        "file://%s/testout1.orc" % test_dataset_tmp,
        streamout,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e.orc" % test_dataset, "%s/testout1.orc_0" % test_dataset_tmp
    )


@pytest.mark.skip_without_hdfs()
def test_hive(vineyard_ipc_socket, vineyard_endpoint, test_dataset, hive_endpoint):
    res = urlparse(hive_endpoint)
    host, port = res.netloc.split(":")
    port = int(port)
    stream = vineyard.io.open(
        "%s:///user/hive/warehouse/pt" % res.scheme,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
    )
    vineyard.io.open(
        "file://%s/testout1.e" % test_dataset,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )


@pytest.mark.skip_without_hdfs()
def test_hdfs_tag(
    vineyard_ipc_socket,
    vineyard_endpoint,
    test_dataset,
    test_dataset_tmp,
    hdfs_endpoint,
):
    res = urlparse(hdfs_endpoint)
    host, port = res.netloc.split(":")
    port = int(port)
    stream = vineyard.io.open(
        "file://%s/ldbc_sample/tag_0_0.csv" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={
            "header_row": True,
            "delimiter": "|",
        },
    )
    vineyard.io.open(
        "%s:///tmp/tag_0_0.csv" % res.scheme,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
    )
    hdfs_stream = vineyard.io.open(
        "%s:///tmp/tag_0_0.csv_0" % res.scheme,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
        read_options={
            "header_row": True,
            "delimiter": "|",
            "schema": "id,name,url",
            "column_types": "int64_t,std::string,std::string",
        },
    )
    vineyard.io.open(
        "file://%s/tag_0_0.out" % test_dataset_tmp,
        hdfs_stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    combine_files("%s/tag_0_0.out" % test_dataset_tmp)
    assert filecmp.cmp(
        "%s/ldbc_sample/tag_0_0.csv" % test_dataset, "%s/tag_0_0.out" % test_dataset_tmp
    )


@pytest.mark.skip_without_hdfs()
def test_hdfs_bytes(
    vineyard_ipc_socket,
    vineyard_endpoint,
    test_dataset,
    test_dataset_tmp,
    hdfs_endpoint,
):
    res = urlparse(hdfs_endpoint)
    host, port = res.netloc.split(":")
    port = int(port)
    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={"header_row": True, "delimiter": " "},
    )
    vineyard.io.open(
        "%s:///tmp/p2p-31.out" % res.scheme,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
    )
    hdfs_stream = vineyard.io.open(
        "%s:///tmp/p2p-31.out_0" % res.scheme,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={"host": host, "port": port},
        read_options={"header_row": True, "delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        hdfs_stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


def test_vineyard_dataframe(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    stream = vineyard.io.open(
        "file://%s/p2p-31.e" % test_dataset,
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={"header_row": False, "delimiter": " "},
    )
    vineyard.io.open(
        "vineyard://p2p-gdf",
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    dfstream = vineyard.io.open(
        "vineyard://p2p-gdf",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        read_options={"delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        dfstream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


@pytest.mark.skip("oss not available at github ci")
def test_oss_read(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")
    stream = vineyard.io.open(
        "oss://grape-uk/p2p-31.e",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "endpoint": endpoint,
        },
        read_options={"header_row": False, "delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


@pytest.mark.skip("oss not available at github ci")
def test_oss_io(vineyard_ipc_socket, vineyard_endpoint):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")
    stream = vineyard.io.open(
        "oss://grape-uk/p2p-31.e",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "endpoint": endpoint,
        },
        read_options={"header_row": False, "delimiter": " "},
        num_workers=2,
    )
    vineyard.io.open(
        "oss://grape-uk/p2p-31.out",
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "endpoint": endpoint,
        },
        num_workers=2,
    )


@pytest.mark.skip("s3 not available at github ci")
def test_s3_read(
    vineyard_ipc_socket, vineyard_endpoint, test_dataset, test_dataset_tmp
):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    region_name = os.environ.get("REGION", "us-east-1")
    stream = vineyard.io.open(
        "s3://ldbc/p2p-31.e",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {"region_name": region_name},
        },
        read_options={"header_row": False, "delimiter": " "},
    )
    vineyard.io.open(
        "file://%s/p2p-31.out" % test_dataset_tmp,
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
    )
    assert filecmp.cmp(
        "%s/p2p-31.e" % test_dataset, "%s/p2p-31.out_0" % test_dataset_tmp
    )


@pytest.mark.skip("s3 not available at github ci")
def test_s3_io(vineyard_ipc_socket, vineyard_endpoint):
    accessKeyID = os.environ["ACCESS_KEY_ID"]
    accessKeySecret = os.environ["SECRET_ACCESS_KEY"]
    region_name = os.environ.get("REGION", "us-east-1")
    stream = vineyard.io.open(
        "s3://ldbc/p2p-31.e",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {"region_name": region_name},
        },
        read_options={"header_row": False, "delimiter": " "},
        num_workers=2,
    )
    vineyard.io.open(
        "s3://ldbc/p2p-31.out",
        stream,
        mode="w",
        vineyard_ipc_socket=vineyard_ipc_socket,
        vineyard_endpoint=vineyard_endpoint,
        storage_options={
            "key": accessKeyID,
            "secret": accessKeySecret,
            "client_kwargs": {"region_name": region_name},
        },
        num_workers=2,
    )


def combine_files(prefix):
    read_files = glob.glob(f"{prefix}_*")
    with open(prefix, "wb") as outfile:
        for f in sorted(read_files):
            with open(f, "rb") as infile:
                outfile.write(infile.read())
