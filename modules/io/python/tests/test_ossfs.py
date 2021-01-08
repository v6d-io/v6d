#! /usr/bin/env python3
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

import datetime
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import chain

import fsspec.core
import oss2
import pytest
import requests

from ossfs import OSSFileSystem

test_bucket_name = "test-bucket-zsy"

files = {
    "test/accounts.1.json": (b'{"amount": 100, "name": "Alice"}\n'
                             b'{"amount": 200, "name": "Bob"}\n'
                             b'{"amount": 300, "name": "Charlie"}\n'
                             b'{"amount": 400, "name": "Dennis"}\n'),
    "test/accounts.2.json": (b'{"amount": 500, "name": "Alice"}\n'
                             b'{"amount": 600, "name": "Bob"}\n'
                             b'{"amount": 700, "name": "Charlie"}\n'
                             b'{"amount": 800, "name": "Dennis"}\n'),
}

csv_files = {
    "2014-01-01.csv": (b"name,amount,id\n"
                       b"Alice,100,1\n"
                       b"Bob,200,2\n"
                       b"Charlie,300,3\n"),
    "2014-01-02.csv": b"name,amount,id\n",
    "2014-01-03.csv": (b"name,amount,id\n"
                       b"Dennis,400,4\n"
                       b"Edith,500,5\n"
                       b"Frank,600,6\n"),
}
text_files = {
    "nested/file1": b"hello\n",
    "nested/file2": b"world",
    "nested/nested2/file1": b"hello\n",
    "nested/nested2/file2": b"world",
}
glob_files = {"file.dat": b"", "filexdat": b""}
a = test_bucket_name + "/tmp/test/a"
b = test_bucket_name + "/tmp/test/b"
c = test_bucket_name + "/tmp/test/c"
d = test_bucket_name + "/tmp/test/d"

fsspec.register_implementation("oss", OSSFileSystem)


@pytest.fixture(scope="session")
def oss():
    key = os.environ.get("ACCESS_KEY_ID")
    secret = os.environ.get("SECRET_ACCESS_KEY")
    endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")
    auth = oss2.Auth(key, secret)
    bucket = oss2.Bucket(auth, endpoint, "test-bucket-zsy")
    try:
        bucket.create_bucket()
    except oss2.exceptions.ServerError:  # bucket exists.
        pass
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
    for flist in [files, csv_files, text_files, glob_files]:
        for f, data in flist.items():
            bucket.put_object(f, data)

    yield oss
    oss._clear_multipart_uploads("test-bucket-zsy")


def test_simple(oss):
    data = b"a" * (10 * 2**20)

    with oss.open(a, "wb") as f:
        f.write(data)

    with oss.open(a, "rb") as f:
        out = f.read(len(data))
        assert len(data) == len(out)
        assert out == data


def test_info(oss):
    oss.touch(a)
    oss.info(a)


def test_info_cached(oss):
    path = test_bucket_name + "/tmp/"
    fqpath = "oss://" + path
    oss.touch(path)
    oss.touch(path + "test")
    info = oss.info(fqpath)
    assert info == oss.info(fqpath)
    assert info == oss.info(path)


def test_seek(oss):
    with oss.open(a, "wb") as f:
        f.write(b"123")

    with oss.open(a) as f:
        f.seek(1000)
        with pytest.raises(ValueError):
            f.seek(-1)
        with pytest.raises(ValueError):
            f.seek(-5, 2)
        with pytest.raises(ValueError):
            f.seek(0, 10)
        f.seek(0)
        assert f.read(1) == b"1"
        f.seek(0)
        assert f.read(1) == b"1"
        f.seek(3)
        assert f.read(1) == b""
        f.seek(-1, 2)
        assert f.read(1) == b"3"
        f.seek(-1, 1)
        f.seek(-1, 1)
        assert f.read(1) == b"2"
        for i in range(4):
            assert f.seek(i) == i


def test_bad_open(oss):
    with pytest.raises(ValueError):
        oss.open("")


def test_read_small(oss):
    fn = test_bucket_name + "/2014-01-01.csv"
    with oss.open(fn, "rb", block_size=10) as f:
        out = []
        while True:
            data = f.read(3)
            if data == b"":
                break
            out.append(data)
        assert oss.cat(fn) == b"".join(out)
        # cache drop
        assert len(f.cache) < len(out)


def test_read_oss_block(oss):
    data = files["test/accounts.1.json"]
    lines = io.BytesIO(data).readlines()
    path = test_bucket_name + "/test/accounts.1.json"
    assert oss.read_block(path, 1, 35, b"\n") == lines[1]
    assert oss.read_block(path, 0, 30, b"\n") == lines[0]
    assert oss.read_block(path, 0, 35, b"\n") == lines[0] + lines[1]
    assert oss.read_block(path, 0, 5000, b"\n") == data
    assert len(oss.read_block(path, 0, 5)) == 5
    assert len(oss.read_block(path, 4, 5000)) == len(data) - 4
    assert oss.read_block(path, 5000, 5010) == b""

    assert oss.read_block(path, 5, None) == oss.read_block(path, 5, 1000)


def test_write_small(oss):
    with oss.open(test_bucket_name + "/test", "wb") as f:
        f.write(b"hello")
    assert oss.cat(test_bucket_name + "/test") == b"hello"
    oss.open(test_bucket_name + "/test", "wb").close()
    assert oss.info(test_bucket_name + "/test")["Size"] == 0


def test_write_large(oss):
    """flush() chunks buffer when processing large singular payload"""
    mb = 2**20
    payload_size = int(2.5 * 5 * mb)
    payload = b"0" * payload_size

    with oss.open(test_bucket_name + "/test", "wb") as fd:
        fd.write(payload)

    assert oss.cat(test_bucket_name + "/test") == payload
    assert oss.info(test_bucket_name + "/test")["Size"] == payload_size


def test_write_limit(oss):
    """flush() respects part_max when processing large singular payload"""
    mb = 2**20
    block_size = 15 * mb
    payload_size = 44 * mb
    payload = b"0" * payload_size

    with oss.open(test_bucket_name + "/test", "wb", blocksize=block_size) as fd:
        fd.write(payload)

    assert oss.cat(test_bucket_name + "/test") == payload

    assert oss.info(test_bucket_name + "/test")["Size"] == payload_size


def test_write_blocks(oss):
    with oss.open(test_bucket_name + "/temp", "wb") as f:
        f.write(b"a" * 2 * 2**20)
        assert f.buffer.tell() == 2 * 2**20
        assert not f.parts
        f.flush()
        assert f.buffer.tell() == 2 * 2**20
        assert not f.parts
        f.write(b"a" * 2 * 2**20)
        f.write(b"a" * 2 * 2**20)
        assert f.mpu
        assert f.parts
    assert oss.info(test_bucket_name + "/temp")["Size"] == 6 * 2**20
    with oss.open(test_bucket_name + "/temp", "wb", block_size=10 * 2**20) as f:
        f.write(b"a" * 15 * 2**20)
        assert f.buffer.tell() == 0
    assert oss.info(test_bucket_name + "/temp")["Size"] == 15 * 2**20


def test_readline(oss):
    all_items = chain.from_iterable([files.items(), csv_files.items(), text_files.items()])
    for k, data in all_items:
        with oss.open("/".join([test_bucket_name, k]), "rb") as f:
            result = f.readline()
            expected = data.split(b"\n")[0] + (b"\n" if data.count(b"\n") else b"")
            assert result == expected


def test_readline_empty(oss):
    data = b""
    with oss.open(a, "wb") as f:
        f.write(data)
    with oss.open(a, "rb") as f:
        result = f.readline()
        assert result == data


def test_readline_blocksize(oss):
    data = b"ab\n" + b"a" * (10 * 2**20) + b"\nab"
    with oss.open(a, "wb") as f:
        f.write(data)
    with oss.open(a, "rb") as f:
        result = f.readline()
        expected = b"ab\n"
        assert result == expected

        result = f.readline()
        expected = b"a" * (10 * 2**20) + b"\n"
        assert result == expected

        result = f.readline()
        expected = b"ab"
        assert result == expected


def test_next(oss):
    expected = csv_files["2014-01-01.csv"].split(b"\n")[0] + b"\n"
    with oss.open(test_bucket_name + "/2014-01-01.csv") as f:
        result = next(f)
        assert result == expected


def test_iterable(oss):
    data = b"abc\n123"
    with oss.open(a, "wb") as f:
        f.write(data)
    with oss.open(a) as f, io.BytesIO(data) as g:
        for fromoss, fromio in zip(f, g):
            assert fromoss == fromio
        f.seek(0)
        assert f.readline() == b"abc\n"
        assert f.readline() == b"123"
        f.seek(1)
        assert f.readline() == b"bc\n"

    with oss.open(a) as f:
        out = list(f)
    with oss.open(a) as f:
        out2 = f.readlines()
    assert out == out2
    assert b"".join(out) == data


def test_readable(oss):
    with oss.open(a, "wb") as f:
        assert not f.readable()

    with oss.open(a, "rb") as f:
        assert f.readable()


def test_seekable(oss):
    with oss.open(a, "wb") as f:
        assert not f.seekable()

    with oss.open(a, "rb") as f:
        assert f.seekable()


def test_writable(oss):
    with oss.open(a, "wb") as f:
        assert f.writable()

    with oss.open(a, "rb") as f:
        assert not f.writable()


def test_append(oss):
    data = text_files["nested/file1"]
    with oss.open(test_bucket_name + "/nested/file1", "ab") as f:
        assert f.tell() == len(data)  # append, no write, small file
    assert oss.cat(test_bucket_name + "/nested/file1") == data
    with oss.open(test_bucket_name + "/nested/file1", "ab") as f:
        f.write(b"extra")  # append, write, small file
    assert oss.cat(test_bucket_name + "/nested/file1") == data + b"extra"

    with oss.open(a, "wb") as f:
        f.write(b"a" * 10 * 2**20)
    with oss.open(a, "ab") as _:
        pass  # append, no write, big file
    assert oss.cat(a) == b"a" * 10 * 2**20

    with oss.open(a, "ab") as f:
        assert f.parts is None
        f._initiate_upload()
        assert f.parts
        assert f.tell() == 10 * 2**20
        f.write(b"extra")  # append, small write, big file
    assert oss.cat(a) == b"a" * 10 * 2**20 + b"extra"

    with oss.open(a, "ab") as f:
        assert f.tell() == 10 * 2**20 + 5
        f.write(b"b" * 10 * 2**20)  # append, big write, big file
        assert f.tell() == 20 * 2**20 + 5
    assert oss.cat(a) == b"a" * 10 * 2**20 + b"extra" + b"b" * 10 * 2**20


def test_bigger_than_block_read(oss):
    with oss.open(test_bucket_name + "/2014-01-01.csv", "rb", block_size=3) as f:
        out = []
        while True:
            data = f.read(20)
            out.append(data)
            if len(data) == 0:
                break
    assert b"".join(out) == csv_files["2014-01-01.csv"]


def test_array(oss):
    from array import array

    data = array("B", [65] * 1000)

    with oss.open(a, "wb") as f:
        f.write(data)

    with oss.open(a, "rb") as f:
        out = f.read()
        assert out == b"A" * 1000


def test_upload_with_ossfs_prefix(oss):
    path = f"oss://{test_bucket_name}/prefix/key"

    with oss.open(path, "wb") as f:
        f.write(b"a" * (10 * 2**20))

    with oss.open(path, "ab") as f:
        f.write(b"b" * (10 * 2**20))


def test_multipart_upload_blocksize(oss):
    blocksize = 5 * (2**20)
    expected_parts = 3

    ossf = oss.open(a, "wb", block_size=blocksize)
    for _ in range(3):
        data = b"b" * blocksize
        ossf.write(data)

    # Ensure that the multipart upload consists of only 3 parts
    assert len(ossf.parts) == expected_parts
    ossf.close()


def test_text_io__stream_wrapper_works(oss):
    """Ensure using TextIOWrapper works."""
    path = f"{test_bucket_name}/file.txt"

    with oss.open(path, "wb") as fd:
        fd.write("\u00af\\_(\u30c4)_/\u00af".encode("utf-16-le"))

    with oss.open(path, "rb") as fd:
        with io.TextIOWrapper(fd, "utf-16-le") as stream:
            assert stream.readline() == "\u00af\\_(\u30c4)_/\u00af"


def test_text_io__basic(oss):
    """Text mode is now allowed."""
    path = f"{test_bucket_name}/file.txt"

    with oss.open(path, "w") as fd:
        fd.write("\u00af\\_(\u30c4)_/\u00af")

    with oss.open(path, "r") as fd:
        assert fd.read() == "\u00af\\_(\u30c4)_/\u00af"


def test_text_io__override_encoding(oss):
    """Allow overriding the default text encoding."""
    path = f"{test_bucket_name}/file.txt"

    with oss.open(path, "w", encoding="ibm500") as fd:
        fd.write("Hello, World!")

    with oss.open(path, "r", encoding="ibm500") as fd:
        assert fd.read() == "Hello, World!"


def test_readinto(oss):
    path = f"{test_bucket_name}/file.txt"

    with oss.open(path, "wb") as fd:
        fd.write(b"Hello, World!")

    contents = bytearray(15)

    with oss.open(path, "rb") as fd:
        assert fd.readinto(contents) == 13

    assert contents.startswith(b"Hello, World!")


def test_autocommit(oss):
    auto_file = test_bucket_name + "/auto_file"
    committed_file = test_bucket_name + "/commit_file"
    aborted_file = test_bucket_name + "/aborted_file"

    def write_and_flush(path, autocommit):
        with oss.open(path, "wb", autocommit=autocommit) as fp:
            fp.write(b"1")
        return fp

    # regular behavior
    fo = write_and_flush(auto_file, autocommit=True)
    assert fo.autocommit
    assert oss.exists(auto_file)

    fo = write_and_flush(committed_file, autocommit=False)
    assert not fo.autocommit
    assert not oss.exists(committed_file)
    fo.commit()
    assert oss.exists(committed_file)

    fo = write_and_flush(aborted_file, autocommit=False)
    assert not oss.exists(aborted_file)
    fo.discard()
    assert not oss.exists(aborted_file)
    # Cannot commit a file that was discarded
    with pytest.raises(Exception):
        fo.commit()


def test_autocommit_mpu(oss):
    """When not autocommitting we always want to use multipart uploads"""
    path = test_bucket_name + "/auto_commit_with_mpu"
    with oss.open(path, "wb", autocommit=False) as fo:
        fo.write(b"1")
    assert fo.mpu is not None
    assert len(fo.parts) == 1
    fo.discard()


def test_seek_reads(oss):
    fn = test_bucket_name + "/myfile"
    with oss.open(fn, "wb") as f:
        f.write(b"a" * 175627146)
    with oss.open(fn, "rb", blocksize=100) as f:
        f.seek(175561610)
        f.read(65536)

        f.seek(4)
        size = 17562198
        d2 = f.read(size)
        assert len(d2) == size

        f.seek(17562288)
        size = 17562187
        d3 = f.read(size)
        assert len(d3) == size
