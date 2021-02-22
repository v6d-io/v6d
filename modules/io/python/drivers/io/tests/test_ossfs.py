#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The file modules/io/python/drivers/io/tests/test_ossfs.py is referred and derived
# from project s3fs,
#     https://github.com/dask/s3fs/blob/main/s3fs/tests/test_s3fs.py
#
#  which has the following license:
#
# Copyright (c) 2016, Continuum Analytics, Inc. and contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# Neither the name of Continuum Analytics nor the names of any contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.

import datetime
import io
import os
import time
import uuid
from itertools import chain

import fsspec.core
import oss2
import oss2.headers
import pytest

ossfs = pytest.importorskip("ossfs")
from ossfs import OSSFileSystem

test_bucket_name = "test-ossfs-zsy"
secure_bucket_name = "test-ossfs-secure-zsy"
versioned_bucket_name = "test-ossfs-versioned-zsy"
tmp_bucket_name = "test-ossfs-tmp-bucket"
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
    "2014-01-02.csv": (b"name,amount,id\n"),
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

key = os.environ.get("ACCESS_KEY_ID")
secret = os.environ.get("SECRET_ACCESS_KEY")
endpoint = os.environ.get("ENDPOINT", "http://oss-cn-hangzhou.aliyuncs.com")

fsspec.register_implementation("oss", OSSFileSystem)


@pytest.fixture(scope="session")
def oss():
    auth = oss2.Auth(key, secret)
    test_bucket = oss2.Bucket(auth, endpoint, test_bucket_name)
    test_bucket.create_bucket(oss2.BUCKET_ACL_PUBLIC_READ)

    versioned_bucket = oss2.Bucket(auth, endpoint, versioned_bucket_name)
    versioned_bucket.create_bucket(oss2.BUCKET_ACL_PUBLIC_READ)
    config = oss2.models.BucketVersioningConfig()
    config.status = oss2.BUCKET_VERSIONING_ENABLE
    versioned_bucket.put_bucket_versioning(config)
    #
    # # secure_bucket = oss2.Bucket(auth, endpoint, secure_bucket_name)
    # # secure_bucket.create_bucket(oss2.BUCKET_ACL_PUBLIC_READ)
    # # try:
    # #     bucket.create_bucket()
    # # except oss2.exceptions.ServerError:  # bucket exists.
    # #     pass
    for flist in [files, csv_files, text_files, glob_files]:
        for f, data in flist.items():
            test_bucket.put_object(f, data)
    OSSFileSystem.clear_instance_cache()
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
    oss.invalidate_cache()
    yield oss
    oss._clear_multipart_uploads(test_bucket_name)
    oss.rm(test_bucket_name, recursive=True)
    oss.rm(versioned_bucket_name, recursive=True)
    try:
        oss.rm(tmp_bucket_name, recursive=True)
    except:
        pass


def test_simple(oss):
    data = b"a" * (10 * 2**20)

    with oss.open(a, "wb") as f:
        f.write(data)

    with oss.open(a, "rb") as f:
        out = f.read(len(data))
        assert len(data) == len(out)
        assert out == data


@pytest.mark.parametrize("default_cache_type", ["none", "bytes", "mmap"])
def test_default_cache_type(oss, default_cache_type):
    data = b"a" * (10 * 2**20)
    oss = OSSFileSystem(
        anon=False,
        key=key,
        secret=secret,
        endpoint=endpoint,
        default_cache_type=default_cache_type,
    )

    with oss.open(a, "wb") as f:
        f.write(data)

    with oss.open(a, "rb") as f:
        assert isinstance(f.cache, fsspec.core.caches[default_cache_type])
        out = f.read(len(data))
        assert len(data) == len(out)
        assert out == data


def test_additional_header():
    oss = OSSFileSystem(anon=True, additional_header={"foo": "bar"})
    assert oss.additional_header.get("foo") == "bar"


def test_additional_params():
    oss = OSSFileSystem(anon=True, additional_params={"foo": "bar"})
    assert oss.additional_params.get("foo") == "bar"


def test_config_kwargs_class_attributes_default():
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
    bucket = oss._make_bucket(test_bucket_name)
    assert bucket.timeout == 60


def test_config_kwargs_class_attributes_override():
    oss = OSSFileSystem(
        key=key,
        secret=secret,
        endpoint=endpoint,
        connect_timeout=120,
    )
    bucket = oss._make_bucket(test_bucket_name)
    assert bucket.timeout == 120


def test_multiple_objects(oss):
    other_oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
    assert oss.ls(f"{test_bucket_name}/test") == other_oss.ls(f"{test_bucket_name}/test")


def test_info(oss):
    oss.touch(a)
    oss.touch(b)
    info = oss.info(a)
    linfo = oss.ls(a, detail=True)[0]
    print(info)
    print(linfo)
    assert abs(info.pop("LastModified") - linfo.pop("LastModified")).seconds < 1
    info.pop("VersionId")
    assert info == linfo
    parent = a.rsplit("/", 1)[0]
    oss.invalidate_cache()  # remove full path from the cache
    oss.ls(parent)  # fill the cache with parent dir
    assert oss.info(a) == oss.dircache[parent][0]  # correct value
    assert id(oss.info(a)) == id(oss.dircache[parent][0])  # is object from cache

    new_parent = test_bucket_name + "/foo"
    oss.mkdir(new_parent)
    with pytest.raises(FileNotFoundError):
        oss.info(new_parent)
    oss.ls(new_parent)
    with pytest.raises(FileNotFoundError):
        oss.info(new_parent)


def test_info_cached(oss):
    path = test_bucket_name + "/tmp/"
    fqpath = "oss://" + path
    oss.touch(path + "test")
    info = oss.info(fqpath)
    assert info == oss.info(fqpath)
    assert info == oss.info(path)


def test_checksum(oss):
    d = "checksum"
    prefix = d + "/e"
    o1 = prefix + "1"
    o2 = prefix + "2"
    path1 = test_bucket_name + "/" + o1
    path2 = test_bucket_name + "/" + o2

    # init client and files
    bucket = oss._make_bucket(test_bucket_name)
    bucket.put_object(o1, "")
    bucket.put_object(o2, "")

    # change one file, using cache
    bucket.put_object(o1, "foo")
    checksum = oss.checksum(path1)
    oss.ls(path1)  # force caching
    bucket.put_object(o1, "bar")
    # refresh == False => checksum doesn't change
    assert checksum == oss.checksum(path1)

    # change one file, without cache
    bucket.put_object(o1, "foo")
    checksum = oss.checksum(path1, refresh=True)
    oss.ls(path1)  # force caching
    bucket.put_object(o1, "bar")
    # refresh == True => checksum changes
    assert checksum != oss.checksum(path1, refresh=True)

    # Test for nonexistent file
    bucket.put_object(o1, "bar")
    oss.ls(path1)  # force caching
    bucket.delete_object(o1)
    with pytest.raises(FileNotFoundError):
        oss.checksum(path1, refresh=True)

    # Test multipart upload
    upload_id = bucket.init_multipart_upload(o1).upload_id
    etag1 = bucket.upload_part(o1, upload_id, 1, "0" * (5 * 1024 * 1024)).etag
    etag2 = bucket.upload_part(o1, upload_id, 2, "0").etag
    parts = [oss2.models.PartInfo(1, etag1), oss2.models.PartInfo(2, etag2)]
    bucket.complete_multipart_upload(o1, upload_id, parts)
    oss.checksum(path1, refresh=True)


test_xattr_sample_metadata = {"x-oss-meta-test-xattr": "1"}


def test_xattr(oss):
    key = "tmp/test/xattr"
    filename = test_bucket_name + "/" + key
    body = b"aaaa"

    bucket: oss2.Bucket = oss._make_bucket(test_bucket_name)
    bucket.put_object(key, body, headers=test_xattr_sample_metadata)
    bucket.put_object_acl(key, oss2.OBJECT_ACL_PUBLIC_READ)

    # save etag for later
    etag = oss.info(filename)["ETag"]
    assert oss2.OBJECT_ACL_PUBLIC_READ == bucket.get_object_acl(key).acl

    assert (oss.getxattr(filename, "x-oss-meta-test-xattr") == test_xattr_sample_metadata["x-oss-meta-test-xattr"])
    assert oss.metadata(filename)["x-oss-meta-test-xattr"] == "1"  # note _ became -

    ossfile = oss.open(filename)
    assert (ossfile.getxattr("x-oss-meta-test-xattr") == test_xattr_sample_metadata["x-oss-meta-test-xattr"])
    assert ossfile.metadata()["x-oss-meta-test-xattr"] == "1"  # note _ became -

    ossfile.setxattr(x_oss_meta_test_xattr="2")
    assert ossfile.getxattr("x-oss-meta-test-xattr") == "2"
    ossfile.setxattr(**{"x-oss-meta-test-xattr": None})
    assert "x-oss-meta-test-xattr" not in ossfile.metadata()
    assert oss.cat(filename) == body

    # check that ACL and ETag are preserved after updating metadata
    assert oss2.OBJECT_ACL_PUBLIC_READ == bucket.get_object_acl(key).acl
    assert oss.info(filename)["ETag"] == etag


def test_xattr_setxattr_in_write_mode(oss):
    ossfile = oss.open(a, "wb")
    with pytest.raises(NotImplementedError):
        ossfile.setxattr(test_xattr="1")


def test_ls(oss):
    assert set(oss.ls("")).issuperset({
        test_bucket_name,
        # secure_bucket_name,
        versioned_bucket_name,
    })
    with pytest.raises(FileNotFoundError):
        oss.ls("nonexistent")
    fn = test_bucket_name + "/test/accounts.1.json"
    assert fn in oss.ls(test_bucket_name + "/test")


def test_pickle(oss):
    import pickle

    oss2 = pickle.loads(pickle.dumps(oss))
    assert oss.ls(test_bucket_name + "/test") == oss2.ls(test_bucket_name + "/test")
    oss3 = pickle.loads(pickle.dumps(oss2))
    assert oss.ls(test_bucket_name + "/test") == oss3.ls(test_bucket_name + "/test")


def test_ls_touch(oss):
    oss.touch(a)
    oss.touch(b)
    L = oss.ls(test_bucket_name + "/tmp/test", True)
    assert {d["Key"] for d in L}.issuperset({a, b})
    L = oss.ls(test_bucket_name + "/tmp/test", False)
    assert set(L).issuperset({a, b})


@pytest.mark.parametrize("version_aware", [True, False])
def test_exists_versioned(oss, version_aware):
    """Test to ensure that a prefix exists when using a versioned bucket"""
    n = 2
    oss = OSSFileSystem(
        key=key,
        secret=secret,
        endpoint=endpoint,
        version_aware=version_aware,
    )
    segments = [versioned_bucket_name] + [str(uuid.uuid4()) for _ in range(n)]
    path = "/".join(segments)
    for i in range(2, n + 1):
        assert not oss.exists("/".join(segments[:i]))
    oss.touch(path)
    for i in range(2, n + 1):
        assert oss.exists("/".join(segments[:i]))


def test_isfile(oss):
    assert not oss.isfile("")
    assert not oss.isfile("/")
    assert not oss.isfile(test_bucket_name)
    assert not oss.isfile(test_bucket_name + "/test")

    assert not oss.isfile(test_bucket_name + "/test/foo")
    assert oss.isfile(test_bucket_name + "/test/accounts.1.json")
    assert oss.isfile(test_bucket_name + "/test/accounts.2.json")
    assert not oss.isfile(a)
    oss.touch(a)
    assert oss.isfile(a)

    assert not oss.isfile(b)
    assert not oss.isfile(b + "/")
    oss.mkdir(b)
    assert not oss.isfile(b)
    assert not oss.isfile(b + "/")

    assert not oss.isfile(c)
    assert not oss.isfile(c + "/")
    oss.mkdir(c + "/")
    assert not oss.isfile(c)
    assert not oss.isfile(c + "/")


def test_isdir(oss):
    assert oss.isdir("")
    assert oss.isdir("/")
    assert oss.isdir(test_bucket_name)
    assert oss.isdir(test_bucket_name + "/test")

    assert not oss.isdir(test_bucket_name + "/test/foo")
    assert not oss.isdir(test_bucket_name + "/test/accounts.1.json")
    assert not oss.isdir(test_bucket_name + "/test/accounts.2.json")

    assert not oss.isdir(a)
    oss.touch(a)
    assert not oss.isdir(a)

    assert not oss.isdir(b)
    assert not oss.isdir(b + "/")

    assert not oss.isdir(c)
    assert not oss.isdir(c + "/")

    # test cache
    oss.invalidate_cache()
    assert not oss.dircache
    oss.ls(test_bucket_name + "/nested")
    assert test_bucket_name + "/nested" in oss.dircache
    assert not oss.isdir(test_bucket_name + "/nested/file1")
    assert not oss.isdir(test_bucket_name + "/nested/file2")
    assert oss.isdir(test_bucket_name + "/nested/nested2")
    assert oss.isdir(test_bucket_name + "/nested/nested2/")


def test_rm(oss):
    oss.touch(a)
    assert oss.exists(a)
    oss.rm(a)
    assert not oss.exists(a)
    # the API is OK with deleting non-files; maybe this is an effect of using bulk
    # with pytest.raises(FileNotFoundError):
    #    oss.rm(test_bucket_name + '/nonexistent')
    with pytest.raises(FileNotFoundError):
        oss.rm("nonexistent")
    oss.rm(test_bucket_name + "/nested", recursive=True)
    assert not oss.exists(test_bucket_name + "/nested/nested2/file1")

    # whole bucket
    bucket = tmp_bucket_name
    oss.mkdir(bucket)
    oss.touch(bucket + "/2014-01-01.csv")
    assert oss.exists(bucket + "/2014-01-01.csv")
    oss.rm(bucket, recursive=True)
    assert not oss.exists(bucket + "/2014-01-01.csv")


def test_rmdir(oss):
    bucket = tmp_bucket_name
    oss.mkdir(bucket)
    assert bucket in oss.ls("/")
    oss.rmdir(bucket)
    assert bucket not in oss.ls("/")


def test_makedirs(oss):
    bucket = tmp_bucket_name
    test_file = bucket + "/a/b/c/file"
    oss.makedirs(test_file)
    assert bucket in oss.ls("/")
    oss.rm(bucket, recursive=True)


def test_bulk_delete(oss):
    with pytest.raises(FileNotFoundError):
        oss.rm(["nonexistent/file"])
    filelist = oss.find(test_bucket_name + "/nested")
    oss.rm(filelist)
    assert not oss.exists(test_bucket_name + "/nested/nested2/file1")


@pytest.mark.skip(reason="anomymous")
def test_anonymous_access(oss):
    oss = OSSFileSystem(anon=True)
    assert oss.ls("") == []
    # TODO: public bucket doesn't work through moto

    with pytest.raises(PermissionError):
        oss.mkdir("newbucket")


def test_oss_file_access(oss):
    fn = test_bucket_name + "/nested/file1"
    data = b"hello\n"
    assert oss.cat(fn) == data
    assert oss.head(fn, 3) == data[:3]
    assert oss.tail(fn, 3) == data[-3:]
    assert oss.tail(fn, 10000) == data


def test_oss_file_info(oss):
    fn = test_bucket_name + "/nested/file1"
    data = b"hello\n"
    assert fn in oss.find(test_bucket_name)
    assert oss.exists(fn)
    assert not oss.exists(fn + "another")
    assert oss.info(fn)["Size"] == len(data)
    with pytest.raises(FileNotFoundError):
        oss.info(fn + "another")


def test_bucket_exists(oss):
    assert oss.exists(test_bucket_name)
    assert not oss.exists(test_bucket_name + "x")


def test_du(oss):
    d = oss.du(test_bucket_name, total=False)
    assert all(isinstance(v, int) and v >= 0 for v in d.values())
    assert test_bucket_name + "/nested/file1" in d

    assert oss.du(test_bucket_name + "/test/", total=True) == sum(map(len, files.values()))
    assert oss.du(test_bucket_name) == oss.du("oss://" + test_bucket_name)


def test_oss_ls(oss):
    fn = test_bucket_name + "/nested/file1"
    assert fn not in oss.ls(test_bucket_name + "/")
    assert fn in oss.ls(test_bucket_name + "/nested/")
    assert fn in oss.ls(test_bucket_name + "/nested")
    assert oss.ls("oss://" + test_bucket_name + "/nested/") == oss.ls(test_bucket_name + "/nested")


def test_oss_big_ls(oss):
    for x in range(1200):
        oss.touch(test_bucket_name + "/thousand/%i.part" % x)
    assert len(oss.find(test_bucket_name)) > 1200
    oss.rm(test_bucket_name + "/thousand/", recursive=True)
    assert len(oss.find(test_bucket_name + "/thousand/")) == 0


def test_oss_ls_detail(oss):
    L = oss.ls(test_bucket_name + "/nested", detail=True)
    assert all(isinstance(item, dict) for item in L)


def test_oss_glob(oss):
    fn = test_bucket_name + "/nested/file1"
    assert fn not in oss.glob(test_bucket_name + "/")
    assert fn not in oss.glob(test_bucket_name + "/*")
    assert fn not in oss.glob(test_bucket_name + "/nested")
    assert fn in oss.glob(test_bucket_name + "/nested/*")
    assert fn in oss.glob(test_bucket_name + "/nested/file*")
    assert fn in oss.glob(test_bucket_name + "/*/*")
    assert all(
        any(p.startswith(f + "/") or p == f for p in oss.find(test_bucket_name))
        for f in oss.glob(test_bucket_name + "/nested/*"))
    assert [test_bucket_name + "/nested/nested2"] == oss.glob(test_bucket_name + "/nested/nested2")
    out = oss.glob(test_bucket_name + "/nested/nested2/*")
    assert {
        f"{test_bucket_name}/nested/nested2/file1",
        f"{test_bucket_name}/nested/nested2/file2",
    } == set(out)

    with pytest.raises(ValueError):
        oss.glob("*")

    # Make sure glob() deals with the dot character (.) correctly.
    assert test_bucket_name + "/file.dat" in oss.glob(test_bucket_name + "/file.*")
    assert test_bucket_name + "/filexdat" not in oss.glob(test_bucket_name + "/file.*")


def test_get_list_of_summary_objects(oss):
    L = oss.ls(test_bucket_name + "/test")

    assert len(L) == 2
    assert [l.lstrip(test_bucket_name).lstrip("/") for l in sorted(L)] == sorted(list(files))

    L2 = oss.ls("oss://" + test_bucket_name + "/test")

    assert L == L2


def test_read_keys_from_bucket(oss):
    for k, data in files.items():
        file_contents = oss.cat("/".join([test_bucket_name, k]))
        assert file_contents == data

        assert oss.cat("/".join([test_bucket_name, k])) == oss.cat("oss://" + "/".join([test_bucket_name, k]))


def test_url(oss):
    fn = test_bucket_name + "/nested/file1"
    url = oss.url(fn, expires=100)
    assert "http" in url
    import urllib.parse

    components = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(components.query)
    exp = int(query["Expires"][0])

    delta = abs(exp - time.time() - 100)
    assert delta < 5

    with oss.open(fn) as f:
        assert "http" in f.url()


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


def test_copy(oss):
    fn = test_bucket_name + "/test/accounts.1.json"
    oss.copy(fn, fn + "2")
    assert oss.cat(fn) == oss.cat(fn + "2")


def test_copy_managed(oss):
    data = b"abc" * 12 * 2**20
    fn = test_bucket_name + "/test/biggerfile"
    with oss.open(fn, "wb") as f:
        f.write(data)
    oss._copy_managed(fn, fn + "2", size=len(data), block=5 * 2**20)
    assert oss.cat(fn) == oss.cat(fn + "2")
    with pytest.raises(ValueError):
        oss._copy_managed(fn, fn + "3", size=len(data), block=4 * 2**20)
    with pytest.raises(ValueError):
        oss._copy_managed(fn, fn + "3", size=len(data), block=6 * 2**30)


@pytest.mark.parametrize("recursive", [True, False])
def test_move(oss, recursive):
    fn = test_bucket_name + "/test/accounts.1.json"
    with oss.open(fn, "wb") as f:
        f.write(files["test/accounts.1.json"])
    data = oss.cat(fn)
    oss.mv(fn, fn + "2", recursive=recursive)
    assert oss.cat(fn + "2") == data
    assert not oss.exists(fn)


def test_get_put(oss, tmpdir):
    test_file = str(tmpdir.join("test.json"))

    oss.get(test_bucket_name + "/test/accounts.1.json", test_file)
    data = files["test/accounts.1.json"]
    assert open(test_file, "rb").read() == data
    oss.put(test_file, test_bucket_name + "/temp")
    assert oss.du(test_bucket_name + "/temp", total=False)[test_bucket_name + "/temp"] == len(data)
    assert oss.cat(test_bucket_name + "/temp") == data


def test_get_put_big(oss, tmpdir):
    test_file = str(tmpdir.join("test"))
    data = b"1234567890A" * 2**20
    open(test_file, "wb").write(data)

    oss.put(test_file, test_bucket_name + "/bigfile")
    test_file = str(tmpdir.join("test2"))
    oss.get(test_bucket_name + "/bigfile", test_file)
    assert open(test_file, "rb").read() == data


@pytest.mark.parametrize("size", [2**10, 2**20, 10 * 2**20])
def test_pipe_cat_big(oss, size):
    data = b"1234567890A" * size
    oss.pipe(test_bucket_name + "/bigfile", data)
    assert oss.cat(test_bucket_name + "/bigfile") == data


def test_errors(oss):
    with pytest.raises(FileNotFoundError):
        oss.open(test_bucket_name + "/tmp/test/shfoshf", "rb")

    # This is fine, no need for interleaving directories on S3
    # with pytest.raises((IOError, OSError)):
    #    oss.touch('tmp/test/shfoshf/x')

    # Deleting nonexistent or zero paths is allowed for now
    # with pytest.raises(FileNotFoundError):
    #    oss.rm(test_bucket_name + '/tmp/test/shfoshf/x')

    with pytest.raises(FileNotFoundError):
        oss.mv(test_bucket_name + "/tmp/test/shfoshf/x", "tmp/test/shfoshf/y")

    with pytest.raises(ValueError):
        oss.open("x", "rb")

    with pytest.raises(FileNotFoundError):
        oss.rm("unknown-ossfs-bucket")

    with pytest.raises(ValueError):
        with oss.open(test_bucket_name + "/temp", "wb") as f:
            f.read()

    with pytest.raises(ValueError):
        f = oss.open(test_bucket_name + "/temp", "rb")
        f.close()
        f.read()

    with pytest.raises(ValueError):
        oss.mkdir("/")

    with pytest.raises(ValueError):
        oss.find("")

    with pytest.raises(ValueError):
        oss.find("oss://")


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


@pytest.mark.xfail(reason="sdk not stable")
def test_new_bucket(oss):
    bucket = tmp_bucket_name
    assert not oss.exists(bucket)
    oss.mkdir(bucket)
    assert oss.exists(bucket)
    with oss.open(f"{bucket}/temp", "wb") as f:
        f.write(b"hello")
    with pytest.raises(OSError):
        oss.rmdir(bucket)

    oss.rm(f"{bucket}/temp")
    oss.rmdir(bucket)
    assert bucket not in oss.ls("")
    assert not oss.exists(bucket)
    with pytest.raises(FileNotFoundError):
        oss.ls(bucket)


@pytest.mark.xfail(reason="sdk not stable")
def test_new_bucket_auto(oss):
    bucket = tmp_bucket_name
    assert not oss.exists(bucket)
    with pytest.raises(Exception):
        oss.mkdir(f"{bucket}/other", create_parents=False)
    oss.mkdir(f"{bucket}/other", create_parents=True)
    assert oss.exists(bucket)
    oss.touch(f"{bucket}/afile")
    with pytest.raises(Exception):
        oss.rm(bucket)
    with pytest.raises(Exception):
        oss.rmdir(bucket)
    oss.rm(bucket, recursive=True)
    assert not oss.exists(bucket)


def test_dynamic_add_rm(oss):
    oss.mkdir(f"{tmp_bucket_name}")
    oss.mkdir(f"{test_bucket_name}/one/two")
    assert oss.exists(f"{test_bucket_name}")
    oss.ls(f"{test_bucket_name}")
    oss.touch(f"{test_bucket_name}/one/two/file_a")
    assert oss.exists(f"{test_bucket_name}/one/two/file_a")
    oss.rm(f"{test_bucket_name}/one", recursive=True)
    assert not oss.exists(f"{test_bucket_name}/one")


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
        assert not (f.parts)
        f.flush()
        assert f.buffer.tell() == 2 * 2**20
        assert not (f.parts)
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


def test_merge(oss):
    with oss.open(a, "wb") as f:
        f.write(b"a" * 10 * 2**20)

    with oss.open(b, "wb") as f:
        f.write(b"a" * 10 * 2**20)
    oss.merge(test_bucket_name + "/joined", [a, b])
    assert oss.info(test_bucket_name + "/joined")["Size"] == 2 * 10 * 2**20


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
    with oss.open(a, "ab") as f:
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


def _get_oss_id(oss):
    return id(oss.oss)


@pytest.mark.xfail()
def test_public_file(oss):
    # works on real oss, not on moto
    test_bucket_name = "ossfs_public_test"
    other_bucket_name = "ossfs_private_test"

    oss.touch(test_bucket_name)
    oss.touch(test_bucket_name + "/afile")
    oss.touch(other_bucket_name, acl="public-read")
    oss.touch(other_bucket_name + "/afile", acl="public-read")

    s = OSSFileSystem(anon=True)
    with pytest.raises(PermissionError):
        s.ls(test_bucket_name)
    s.ls(other_bucket_name)

    oss.chmod(test_bucket_name, acl="public-read")
    oss.chmod(other_bucket_name, acl="private")
    with pytest.raises(PermissionError):
        s.ls(other_bucket_name, refresh=True)
    assert s.ls(test_bucket_name, refresh=True)

    # public file in private bucket
    with oss.open(other_bucket_name + "/see_me", "wb", acl="public-read") as f:
        f.write(b"hello")
    assert s.cat(other_bucket_name + "/see_me") == b"hello"


def test_upload_with_ossfs_prefix(oss):
    path = f"oss://{test_bucket_name}/test/prefix/key"

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


def test_default_pars(oss):
    oss = OSSFileSystem(
        key=key,
        secret=secret,
        endpoint=endpoint,
        default_block_size=20,
        default_fill_cache=False,
    )
    fn = test_bucket_name + "/" + list(files)[0]
    with oss.open(fn) as f:
        assert f.blocksize == 20
        assert f.fill_cache is False
    with oss.open(fn, block_size=40, fill_cache=True) as f:
        assert f.blocksize == 40
        assert f.fill_cache is True


def test_tags(oss):
    tagset = {"tag1": "value1", "tag2": "value2"}
    fname = test_bucket_name + "/" + list(files)[0]
    oss.touch(fname)
    oss.put_tags(fname, tagset)
    assert oss.get_tags(fname) == tagset

    # Ensure merge mode updates value of existing key and adds new one
    new_tagset = {"tag2": "updatedvalue2", "tag3": "value3"}
    oss.put_tags(fname, new_tagset, mode="m")
    tagset.update(new_tagset)
    assert oss.get_tags(fname) == tagset


def test_versions(oss):
    versioned_file = versioned_bucket_name + "/versioned_file_" + str(uuid.uuid4())
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)

    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"1")
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"2")
    assert oss.isfile(versioned_file)
    versions = oss.object_version_info(versioned_file)
    version_ids = [version.versionid for version in reversed(versions)]
    assert len(version_ids) == 2
    with oss.open(versioned_file) as fo:
        assert fo.version_id == version_ids[1]
        assert fo.read() == b"2"

    with oss.open(versioned_file, version_id=version_ids[0]) as fo:
        assert fo.version_id == version_ids[0]
        assert fo.read() == b"1"


def test_list_versions_many(oss):
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)
    versioned_file = versioned_bucket_name + "/versioned_file-" + str(uuid.uuid4())
    for i in range(1200):
        with oss.open(versioned_file, "wb") as fo:
            fo.write(b"1")
    versions = oss.object_version_info(versioned_file)
    assert len(versions) == 1200


def test_fsspec_versions_multiple(oss):
    """Test that the standard fsspec.core.get_fs_token_paths behaves as expected for versionId urls"""
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)
    versioned_file = versioned_bucket_name + "/versioned_file3-" + str(uuid.uuid4())
    version_lookup = {}
    for i in range(20):
        contents = str(i).encode()
        with oss.open(versioned_file, "wb") as fo:
            fo.write(contents)
        version_lookup[fo.version_id] = contents
    urls = ["oss://{}?versionId={}".format(versioned_file, version) for version in version_lookup.keys()]
    fs, token, paths = fsspec.core.get_fs_token_paths(urls,
                                                      storage_options=dict(key=key, secret=secret, endpoint=endpoint))
    assert isinstance(fs, OSSFileSystem)
    assert fs.version_aware
    for path in paths:
        with fs.open(path, "rb") as fo:
            contents = fo.read()
            assert contents == version_lookup[fo.version_id]


def test_versioned_file_fullpath(oss):
    versioned_file = (versioned_bucket_name + "/versioned_file_fullpath-" + str(uuid.uuid4()))
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"1")
    # moto doesn't correctly return a versionId for a multipart upload. So we resort to this.
    # version_id = fo.version_id
    versions = oss.object_version_info(versioned_file)
    version_ids = [version.versionid for version in reversed(versions)]
    version_id = version_ids[0]

    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"2")

    file_with_version = "{}?versionId={}".format(versioned_file, version_id)

    with oss.open(file_with_version, "rb") as fo:
        assert fo.version_id == version_id
        assert fo.read() == b"1"


def test_versions_unaware(oss):
    versioned_file = versioned_bucket_name + "/versioned_file3"
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=False)
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"1")
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"2")

    with oss.open(versioned_file) as fo:
        assert fo.version_id is None
        assert fo.read() == b"2"

    with pytest.raises(ValueError):
        with oss.open(versioned_file, version_id="0"):
            fo.read()


def test_text_io__stream_wrapper_works(oss):
    """Ensure using TextIOWrapper works."""
    with oss.open(f"{test_bucket_name}/file.txt", "wb") as fd:
        fd.write("\u00af\\_(\u30c4)_/\u00af".encode("utf-16-le"))

    with oss.open(f"{test_bucket_name}/file.txt", "rb") as fd:
        with io.TextIOWrapper(fd, "utf-16-le") as stream:
            assert stream.readline() == "\u00af\\_(\u30c4)_/\u00af"


def test_text_io__basic(oss):
    """Text mode is now allowed."""
    with oss.open(f"{test_bucket_name}/file.txt", "w") as fd:
        fd.write("\u00af\\_(\u30c4)_/\u00af")

    with oss.open(f"{test_bucket_name}/file.txt", "r") as fd:
        assert fd.read() == "\u00af\\_(\u30c4)_/\u00af"


def test_text_io__override_encoding(oss):
    """Allow overriding the default text encoding."""
    with oss.open(f"{test_bucket_name}/file.txt", "w", encoding="ibm500") as fd:
        fd.write("Hello, World!")

    with oss.open(f"{test_bucket_name}/file.txt", "r", encoding="ibm500") as fd:
        assert fd.read() == "Hello, World!"


def test_readinto(oss):
    with oss.open(f"{test_bucket_name}/file.txt", "wb") as fd:
        fd.write(b"Hello, World!")

    contents = bytearray(15)

    with oss.open(f"{test_bucket_name}/file.txt", "rb") as fd:
        assert fd.readinto(contents) == 13

    assert contents.startswith(b"Hello, World!")


def test_change_defaults_only_subsequent(oss):
    """Test for Issue #135

    Ensure that changing the default block size doesn't affect existing file
    systems that were created using that default. It should only affect file
    systems created after the change.
    """
    try:
        OSSFileSystem.cachable = False  # don't reuse instances with same pars

        fs_default = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
        assert fs_default.default_block_size == 5 * (1024**2)

        fs_overridden = OSSFileSystem(
            default_block_size=64 * (1024**2),
            key=key,
            secret=secret,
            endpoint=endpoint,
        )
        assert fs_overridden.default_block_size == 64 * (1024**2)

        # Suppose I want all subsequent file systems to have a block size of 1 GiB
        # instead of 5 MiB:
        OSSFileSystem.default_block_size = 1024**3

        fs_big = OSSFileSystem(key=key, secret=secret, endpoint=endpoint)
        assert fs_big.default_block_size == 1024**3

        # Test the other file systems created to see if their block sizes changed
        assert fs_overridden.default_block_size == 64 * (1024**2)
        assert fs_default.default_block_size == 5 * (1024**2)
    finally:
        OSSFileSystem.default_block_size = 5 * (1024**2)
        OSSFileSystem.cachable = True


def test_cache_after_copy(oss):
    # https://github.com/dask/dask/issues/5134
    prefix = f"{test_bucket_name}/test"
    oss.touch(f"{prefix}/afile")
    assert f"{prefix}/afile" in oss.ls(f"oss://{prefix}", False)
    oss.cp(f"{prefix}/afile", f"{prefix}/bfile")
    assert f"{prefix}/bfile" in oss.ls(f"oss://{prefix}", False)


def test_autocommit(oss):
    auto_file = test_bucket_name + "/auto_file"
    committed_file = test_bucket_name + "/commit_file"
    aborted_file = test_bucket_name + "/aborted_file"
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)

    def write_and_flush(path, autocommit):
        with oss.open(path, "wb", autocommit=autocommit) as fo:
            fo.write(b"1")
        return fo

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


def test_touch(oss):
    # create
    fn = test_bucket_name + "/touched"
    assert not oss.exists(fn)
    oss.touch(fn)
    assert oss.exists(fn)
    assert oss.size(fn) == 0

    # truncates
    with oss.open(fn, "wb") as f:
        f.write(b"data")
    assert oss.size(fn) == 4
    oss.touch(fn, truncate=True)
    assert oss.size(fn) == 0

    # exists error
    with oss.open(fn, "wb") as f:
        f.write(b"data")
    assert oss.size(fn) == 4
    with pytest.raises(ValueError):
        oss.touch(fn, truncate=False)
    assert oss.size(fn) == 4


def test_touch_versions(oss):
    versioned_file = versioned_bucket_name + "/versioned_file-" + str(uuid.uuid4())
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)
    returned_versions = []
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"1")
    returned_versions.append(fo.version_id)
    with oss.open(versioned_file, "wb") as fo:
        fo.write(b"")
    returned_versions.append(fo.version_id)
    assert oss.isfile(versioned_file)
    versions = oss.object_version_info(versioned_file)
    version_ids = [version.versionid for version in reversed(versions)]
    assert len(version_ids) == 2

    with oss.open(versioned_file) as fo:
        assert fo.version_id == version_ids[1]
        assert fo.version_id == returned_versions[1]
        assert fo.read() == b""

    with oss.open(versioned_file, version_id=version_ids[0]) as fo:
        assert fo.version_id == version_ids[0]
        assert fo.version_id == returned_versions[0]
        assert fo.read() == b"1"


def test_cat_missing(oss):
    fn0 = test_bucket_name + "/file0"
    fn1 = test_bucket_name + "/file1"
    oss.touch(fn0)
    with pytest.raises(FileNotFoundError):
        oss.cat([fn0, fn1], on_error="raise")
    out = oss.cat([fn0, fn1], on_error="omit")
    assert list(out) == [fn0]
    out = oss.cat([fn0, fn1], on_error="return")
    assert fn1 in out
    assert isinstance(out[fn1], FileNotFoundError)


def test_get_directories(oss, tmpdir):
    oss.touch(test_bucket_name + "/dir/dirkey/key0")
    oss.touch(test_bucket_name + "/dir/dirkey/key1")
    oss.touch(test_bucket_name + "/dir/dirkey")
    oss.touch(test_bucket_name + "/dir/dir/key")
    d = str(tmpdir)
    oss.get(test_bucket_name + "/dir", d, recursive=True)
    assert {"dirkey", "dir"} == set(os.listdir(d))
    assert ["key"] == os.listdir(os.path.join(d, "dir"))
    assert {"key0", "key1"} == set(os.listdir(os.path.join(d, "dirkey")))


def test_seek_reads(oss):
    fn = test_bucket_name + "/myfile"
    with oss.open(fn, "wb") as f:
        f.write(b"a" * 175627146)
    with oss.open(fn, "rb", blocksize=100) as f:
        f.seek(175561610)
        d1 = f.read(65536)

        f.seek(4)
        size = 17562198
        d2 = f.read(size)
        assert len(d2) == size

        f.seek(17562288)
        size = 17562187
        d3 = f.read(size)
        assert len(d3) == size


def test_connect_many(oss):
    from multiprocessing.pool import ThreadPool

    def task(i):
        OSSFileSystem(key=key, secret=secret, endpoint=endpoint).ls("")
        return True

    pool = ThreadPool(processes=20)
    out = pool.map(task, range(40))
    assert all(out)
    pool.close()
    pool.join()


def test_requester_pays(oss):
    fn = test_bucket_name + "/myfile"
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, requester_pays=True)
    assert oss.additional_header[oss2.headers.OSS_REQUEST_PAYER] == "requester"
    oss.mkdir(test_bucket_name)
    oss.touch(fn)
    with oss.open(fn, "rb") as f:
        assert f.oss.additional_header[oss2.headers.OSS_REQUEST_PAYER] == "requester"


def test_modified(oss):
    dir_path = test_bucket_name + "/modified"
    file_path = dir_path + "/file"

    # Test file
    oss.touch(file_path)
    modified = oss.modified(path=file_path)
    assert isinstance(modified, datetime.datetime)

    # Test directory
    with pytest.raises(IsADirectoryError):
        modified = oss.modified(path=dir_path)

    # Test bucket
    with pytest.raises(IsADirectoryError):
        oss.modified(path=test_bucket_name)


def test_via_fsspec(oss):
    import fsspec

    oss.mkdir(test_bucket_name + "/mine")
    with fsspec.open(test_bucket_name + "/mine/oi", "wb") as f:
        f.write(b"hello")
    with fsspec.open(test_bucket_name + "/mine/oi", "rb") as f:
        assert f.read() == b"hello"


def test_repeat_exists(oss):
    fn = "oss://" + test_bucket_name + "/file1"
    oss.touch(fn)

    assert oss.exists(fn)
    assert oss.exists(fn)


def test_with_xzarr(oss):
    da = pytest.importorskip("dask.array")
    xr = pytest.importorskip("xarray")
    name = "sample"

    nana = xr.DataArray(da.random.random((1024, 1024, 10, 9, 1)))

    oss_path = f"{test_bucket_name}/{name}"
    ossstore = oss.get_mapper(oss_path)

    oss.ls("")
    nana.to_dataset().to_zarr(store=ossstore, mode="w", consolidated=True, compute=True)


def test_shallow_find(oss):
    """Test that find method respects maxdepth.

    Verify that the ``find`` method respects the ``maxdepth`` parameter.  With
    ``maxdepth=1``, the results of ``find`` should be the same as those of
    ``ls``, without returning subdirectories.  See also issue 378.
    """

    assert oss.ls(test_bucket_name) == oss.find(test_bucket_name, maxdepth=1, withdirs=True)
    assert oss.ls(test_bucket_name) == oss.glob(test_bucket_name + "/*")


def test_version_sizes(oss):
    # protect against caching of incorrect version details
    oss = OSSFileSystem(key=key, secret=secret, endpoint=endpoint, version_aware=True)
    import gzip

    path = f"oss://{versioned_bucket_name}/test.txt.gz"
    versions = [
        oss.pipe_file(path, gzip.compress(text)) for text in (
            b"good morning!",
            b"hello!",
            b"hi!",
            b"hello!",
        )
    ]
    for version in versions:
        version_id = version.versionid
        with oss.open(path, version_id=version_id) as f:
            with gzip.open(f) as zfp:
                zfp.read()


def test_find_no_side_effect(oss):
    infos1 = oss.find(test_bucket_name, maxdepth=1, withdirs=True, detail=True)
    oss.find(test_bucket_name, maxdepth=None, withdirs=True, detail=True)
    infooss = oss.find(test_bucket_name, maxdepth=1, withdirs=True, detail=True)
    assert infos1.keys() == infooss.keys()


def test_get_file_info_with_selector(oss):
    fs = oss
    base_dir = test_bucket_name + "/selector-dir/"
    file_a = base_dir + "test_file_a"
    file_b = base_dir + "test_file_b"
    dir_a = base_dir + "test_dir_a"
    file_c = base_dir + "test_dir_a/test_file_c"

    try:
        fs.mkdir(base_dir)
        with fs.open(file_a, mode="wb"):
            pass
        with fs.open(file_b, mode="wb"):
            pass
        fs.mkdir(dir_a)
        with fs.open(file_c, mode="wb"):
            pass

        infos = fs.find(base_dir, maxdepth=None, withdirs=True, detail=True)
        assert len(infos) == 4

        for info in infos.values():
            if info["name"].endswith(file_a):
                assert info["type"] == "file"
            elif info["name"].endswith(file_b):
                assert info["type"] == "file"
            elif info["name"].endswith(file_c):
                assert info["type"] == "file"
            elif info["name"].rstrip("/").endswith(dir_a):
                assert info["type"] == "directory"
            else:
                raise ValueError("unexpected path {}".format(info["name"]))
    finally:
        fs.rm(base_dir, recursive=True)
