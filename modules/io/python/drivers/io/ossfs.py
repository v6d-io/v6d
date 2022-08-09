#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The file modules/io/python/drivers/io/ossfs.py is referred and derived
# from project s3fs,
#     https://github.com/dask/s3fs/blob/main/s3fs/core.py
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
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

# pylint: skip-file

import itertools
import logging
import os
import re
import socket
import time
from datetime import datetime
from typing import Optional
from typing import Tuple
from urllib.parse import urlsplit

import oss2
from fsspec.spec import AbstractBufferedFile
from fsspec.spec import AbstractFileSystem
from fsspec.utils import tokenize
from fsspec.utils import update_storage_options
from oss2 import exceptions
from oss2.models import PartInfo

logger = logging.getLogger("vineyard.io.ossfs")


def setup_logging(level=None):
    handle = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s " "- %(message)s"
    )
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(level or os.environ["OSSFS_LOGGING_LEVEL"])


if "OSSFS_LOGGING_LEVEL" in os.environ:
    setup_logging()

RETRYABLE_ERRORS = (socket.timeout,)

_VALID_FILE_MODES = {"r", "w", "a", "rb", "wb", "ab"}

key_acls = {
    "private": oss2.OBJECT_ACL_PRIVATE,
    "public-read": oss2.OBJECT_ACL_PUBLIC_READ,
    "public-read-write": oss2.OBJECT_ACL_PUBLIC_READ_WRITE,
    "default": oss2.OBJECT_ACL_DEFAULT,
}

buck_acls = {
    "private": oss2.BUCKET_ACL_PRIVATE,
    "public-read": oss2.BUCKET_ACL_PUBLIC_READ,
    "public-read-write": oss2.BUCKET_ACL_PUBLIC_READ_WRITE,
}


def infer_storage_options(urlpath, inherit_storage_options=None):
    # Handle Windows paths including disk name in this special case
    if (
        re.match(r"^[a-zA-Z]:[\\/]", urlpath)
        or re.match(r"^[a-zA-Z0-9]+://", urlpath) is None
    ):
        return {"protocol": "file", "path": urlpath}

    parsed_path = urlsplit(urlpath)
    protocol = parsed_path.scheme or "file"
    if parsed_path.fragment:
        path = "#".join([parsed_path.path, parsed_path.fragment])
    else:
        path = parsed_path.path
    if protocol == "file":
        # Special case parsing file protocol URL on Windows according to:
        # https://msdn.microsoft.com/en-us/library/jj710207.aspx
        windows_path = re.match(r"^/([a-zA-Z])[:|]([\\/].*)$", path)
        if windows_path:
            path = "%s:%s" % windows_path.groups()

    if protocol in ["http", "https"]:
        # for HTTP, we don't want to parse, as requests will anyway
        return {"protocol": protocol, "path": urlpath}

    options = {"protocol": protocol, "path": path}

    if parsed_path.netloc:
        # Parse `hostname` from netloc manually because `parsed_path.hostname`
        # lowercases the hostname which is not always desirable (e.g. in S3):
        # https://github.com/dask/dask/issues/1417
        options["host"] = parsed_path.netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0]

        if protocol in ("s3", "gcs", "gs", "oss"):
            options["path"] = options["host"] + options["path"]
        else:
            options["host"] = options["host"]
        if parsed_path.port:
            options["port"] = parsed_path.port
        if parsed_path.username:
            options["username"] = parsed_path.username
        if parsed_path.password:
            options["password"] = parsed_path.password

    if parsed_path.query:
        options["url_query"] = parsed_path.query
    if parsed_path.fragment:
        options["url_fragment"] = parsed_path.fragment

    if inherit_storage_options:
        update_storage_options(options, inherit_storage_options)

    return options


def version_id_kw(version_id):

    return {"versionId": version_id} if version_id else {}


def _coalesce_version_id(*args):
    """Helper to coalesce a list of version_ids down to one"""
    version_ids = set(args)
    if None in version_ids:
        version_ids.remove(None)
    if len(version_ids) > 1:
        raise ValueError(
            "Cannot coalesce version_ids where more than one are defined,"
            " {}".format(version_ids)
        )
    elif len(version_ids) == 0:
        return None
    else:
        return version_ids.pop()


def _get_brange(size, block):
    """
    Chunk up a file into zero-based byte ranges

    Parameters
    ----------
    size : file size
    block : block size
    """
    for offset in range(0, size, block):
        yield offset, min(offset + block - 1, size - 1)


class OSSFileSystem(AbstractFileSystem):
    root_marker = ""
    connect_timeout = 60
    retries = 5
    default_block_size = 5 * 2**20
    protocol = ["oss"]
    _extra_tokenize_attributes = ("default_block_size",)

    def __init__(
        self,
        anon=False,
        key=None,
        secret=None,
        token=None,
        endpoint=None,
        is_cname=False,
        session=None,
        connect_timeout=None,
        app_name="",
        enable_crc=True,
        requester_pays=False,
        default_block_size=None,
        default_fill_cache=True,
        default_cache_type="bytes",
        version_aware=False,
        additional_header=None,
        additional_params=None,
        username=None,
        password=None,
        **kwargs
    ):
        if key and username:
            raise KeyError("Supply either key or username, not both")
        if secret and password:
            raise KeyError("Supply secret or password, not both")
        if username:
            key = username
        if password:
            secret = password

        # For auth and bucket
        self.anon = anon
        self.key = key
        self.secret = secret
        self.token = token
        self.endpoint = endpoint
        self.is_cname = is_cname
        self.session = session
        if connect_timeout is not None:
            self.connect_timeout = connect_timeout
        self.app_name = app_name
        self.enable_crc = enable_crc
        self.kwargs = kwargs

        # passed to fsspec superclass
        super_kwargs = {
            k: kwargs.pop(k)
            for k in ["use_listings_cache", "listings_expiry_time", "max_paths"]
            if k in kwargs
        }
        super().__init__(loop=None, asynchronous=False, **super_kwargs)

        self.default_block_size = default_block_size or self.default_block_size
        self.default_fill_cache = default_fill_cache
        self.default_cache_type = default_cache_type
        self.version_aware = version_aware
        self.additional_header = additional_header or {}
        self.additional_params = additional_params or {}
        if requester_pays:
            self.additional_header[oss2.headers.OSS_REQUEST_PAYER] = "requester"
        if anon:
            self.auth = oss2.AnonymousAuth()
        elif token:
            self.auth = oss2.StsAuth(key, secret, token)
        else:
            self.auth = oss2.Auth(key, secret)

    def _make_bucket(self, bucket_name):
        return oss2.Bucket(
            self.auth,
            self.endpoint,
            bucket_name,
            self.is_cname,
            self.session,
            self.connect_timeout,
            self.app_name,
            self.enable_crc,
        )

    def _make_service(self):
        return oss2.Service(
            self.auth, self.endpoint, self.session, self.connect_timeout, self.app_name
        )

    def call_oss(self, bucket_name, method, *args, **kwargs):
        kw2 = kwargs.copy()
        logger.debug("CALL: %s - %s - %s - %s", bucket_name, method, args, kw2)
        bucket = self._make_bucket(bucket_name)
        caller = getattr(bucket, method)
        for i in range(self.retries):
            try:
                return caller(*args, **kwargs)
            except RETRYABLE_ERRORS as e:
                logger.debug("Retryable error: %s", e)
                time.sleep(min(1.7**i * 0.1, 15))
            except Exception as e:
                logger.exception("Nonretryable error: %s", e)
                raise

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        """
        When we have a urlpath that contains a ?versionId=

        Assume that we want to use version_aware mode for
        the filesystem.
        """
        url_storage_opts = infer_storage_options(urlpath)
        url_query = url_storage_opts.get("url_query")
        out = {}
        if url_query is not None:
            from urllib.parse import parse_qs

            parsed = parse_qs(url_query)
            if "versionId" in parsed:
                out["version_aware"] = True
        return out

    def split_path(self, path) -> Tuple[str, str, Optional[str]]:
        """Normalize path string into bucket and key, and optionally, version_id."""
        path = self._strip_protocol(path)
        path = path.lstrip("/")
        if "/" not in path:
            return path, "", None
        else:
            bucket, keypart = path.split("/", 1)
            key, _, version_id = keypart.partition("?versionId=")
            return (
                bucket,
                key,
                version_id if self.version_aware and version_id else None,
            )

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        version_id=None,
        fill_cache=None,
        cache_type=None,
        autocommit=True,
        **kwargs
    ):
        if block_size is None:
            block_size = self.default_block_size
        if fill_cache is None:
            fill_cache = self.default_fill_cache

        if not self.version_aware and version_id:
            raise ValueError(
                "version_id cannot be specified if the filesystem "
                "is not version aware"
            )

        if cache_type is None:
            cache_type = self.default_cache_type

        return OSSFile(
            self,
            path,
            mode,
            block_size=block_size,
            version_id=version_id,
            fill_cache=fill_cache,
            cache_type=cache_type,
            autocommit=autocommit,
        )

    def find(self, path, maxdepth=None, withdirs=None, detail=False):
        bucket_name, key, _ = self.split_path(path)
        if not bucket_name:
            raise ValueError("Cannot traverse all of OSS")
        if maxdepth:
            return super().find(
                bucket_name + "/" + key,
                maxdepth=maxdepth,
                withdirs=withdirs,
                detail=detail,
            )

        out = self._lsdir(path, delimiter="")
        if not out and key:
            try:
                out = [self._info(path)]
            except FileNotFoundError:
                out = []
        dirs = []
        sdirs = set()
        for o in out:
            par = self._parent(o["name"])
            if par not in self.dircache:
                if par not in sdirs:
                    sdirs.add(par)
                    if len(path) <= len(par):
                        dirs.append(
                            {
                                "Key": self.split_path(par)[1],
                                "Size": 0,
                                "name": par,
                                "StorageClass": "DIRECTORY",
                                "type": "directory",
                                "size": 0,
                            }
                        )
                    self.dircache[par] = []
            if par in sdirs:
                self.dircache[par].append(o)
        if withdirs:
            out = sorted(out + dirs, key=lambda x: x["name"])
        if detail:
            return {o["name"]: o for o in out}
        return [o["name"] for o in out]

    def mkdir(self, path, acl="", create_parents=True, **kwargs):
        path = self._strip_protocol(path).rstrip("/")
        bucket_name, key, _ = self.split_path(path)
        if not key or (create_parents and not self.exists(bucket_name)):
            if acl and acl not in buck_acls:
                raise ValueError("ACL not in %s", buck_acls)
            if not acl:
                acl = "public-read"
            try:
                self.call_oss(bucket_name, "create_bucket", buck_acls[acl])
                self.invalidate_cache("")
                self.invalidate_cache(bucket_name)
            except exceptions.ClientError as e:
                raise ValueError("Bucket create failed %r: %s" % (bucket_name, e))
        else:
            # raises if bucket doesn't exist, but doesn't write anything
            self._ls(bucket_name)

    def makedirs(self, path, exist_ok=False):
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if exist_ok:
                pass
            else:
                raise

    def rmdir(self, path):
        try:
            self.call_oss(path, "delete_bucket")
        except oss2.exceptions.BucketNotEmpty as e:
            raise OSError from e
        except oss2.exceptions.NoSuchBucket as e:
            raise FileNotFoundError(path) from e

        self.invalidate_cache(path)
        self.invalidate_cache("")

    def exists(self, path, **kwargs):
        if path in ["", "/"]:
            # the root always exists, even if anon
            return True
        bucket_name, key, version_id = self.split_path(path)

        if key:
            try:
                if self._ls_from_cache(path):
                    return True
            except FileNotFoundError:
                return False
            try:
                self._info(path)
                return True
            except FileNotFoundError:
                return False
        elif self.dircache.get(bucket_name, False):
            return True
        else:
            try:
                if self._ls_from_cache(bucket_name):
                    return True
            except FileNotFoundError:
                # might still be a bucket we can access but don't own
                pass
            try:
                self.call_oss(bucket_name, "get_bucket_info")
                return True
            except Exception:
                return False

    def touch(self, path, truncate=True, data=None, **kwargs):
        """Create empty file or truncate"""
        bucket_name, key, version_id = self.split_path(path)
        if version_id:
            raise ValueError("OSS does not support touching existing versions of files")
        if not truncate and self.exists(path):
            raise ValueError("OSS does not support touching existent files")
        try:
            kwargs.update(self.additional_header)
            write_result = self.call_oss(
                bucket_name, "put_object", key, data="", headers=kwargs
            )
        except Exception as ex:
            raise ex
        self.invalidate_cache(self._parent(path))
        return write_result

    def cat_file(self, path, version_id=None, start=None, end=None):
        bucket_name, key, vers = self.split_path(path)
        if (start is None) ^ (end is None):
            raise ValueError("Give start and end or neither")
        if start is not None and end is not None:
            byte_range = (start, end)
        else:
            byte_range = None
        params = version_id_kw(version_id or vers)
        try:
            stream = self.call_oss(
                bucket_name,
                "get_object",
                key,
                byte_range,
                headers=self.additional_header,
                params=params,
            )
        except exceptions.NotFound:
            raise FileNotFoundError(path)
        data = stream.read()
        stream.close()
        return data

    def pipe_file(self, path, data, chunksize=50 * 2**20, **kwargs):
        bucket_name, key, _ = self.split_path(path)
        size = len(data)
        # Max size of PutObject is 5GB
        if size < min(5 * 2**30, 2 * chunksize):
            return self.call_oss(bucket_name, "put_object", key, data, **kwargs)
        else:
            mpu = self.call_oss(bucket_name, "init_multipart_upload", key, **kwargs)

            out = [
                self.call_oss(
                    bucket_name,
                    "upload_part",
                    key,
                    mpu.upload_id,
                    i + 1,
                    data[off : off + chunksize],
                )
                for i, off in enumerate(range(0, len(data), chunksize))
            ]

            parts = [PartInfo(i + 1, o.etag) for i, o in enumerate(out)]
            self.call_oss(
                bucket_name, "complete_multipart_upload", key, mpu.upload_id, parts
            )
        self.invalidate_cache(path)

    def put_file(self, lpath, rpath, chunksize=50 * 2**20, **kwargs):
        bucket_name, key, _ = self.split_path(rpath)
        if os.path.isdir(lpath) and key:
            # don't make remote "directory"
            return
        size = os.path.getsize(lpath)
        with open(lpath, "rb") as f0:
            if size < min(5 * 2**30, 2 * chunksize):
                return self.call_oss(bucket_name, "put_object", key, f0, **kwargs)
            else:
                mpu = self.call_oss(bucket_name, "init_multipart_upload", key, **kwargs)
                out = []
                while True:
                    chunk = f0.read(chunksize)
                    if not chunk:
                        break
                    out.append(
                        self.call_oss(
                            bucket_name,
                            "upload_part",
                            key,
                            mpu.upload_id,
                            len(out) + 1,
                            chunk,
                        )
                    )

                parts = [PartInfo(i + 1, o.etag) for i, o in enumerate(out)]
                self.call_oss(
                    bucket_name, "complete_multipart_upload", key, mpu.upload_id, parts
                )
        self.invalidate_cache(rpath)

    def get_file(self, rpath, lpath, version_id=None):
        bucket_name, key, vers = self.split_path(rpath)
        if os.path.isdir(lpath):
            return
        if self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return
        self.call_oss(
            bucket_name,
            "get_object_to_file",
            key,
            lpath,
            params=version_id_kw(version_id or vers),
            headers=self.additional_header,
        )

    def _info(self, path, bucket_name=None, key=None, kwargs={}, version_id=None):
        if bucket_name is None:
            bucket_name, key, version_id = self.split_path(path)

        try:
            out = self.call_oss(
                bucket_name,
                "head_object",
                key,
                params=version_id_kw(version_id),
                headers=self.additional_header,
            )
            return {
                "ETag": out.etag,
                "Key": "/".join([bucket_name, key]),
                "LastModified": datetime.fromtimestamp(out.last_modified),
                "Size": out.content_length,
                "size": out.content_length,
                "name": "/".join([bucket_name, key]),
                "type": "file",
                "StorageClass": "STANDARD",
                "VersionId": out.versionid,
            }
        except exceptions.NoSuchBucket:
            raise FileNotFoundError(path)
        except exceptions.NotFound:
            pass

        try:
            # We check to see if the path is a directory by attempting to list its
            # contexts. If anything is found, it is indeed a directory
            out = self.call_oss(
                bucket_name,
                "list_objects_v2",
                key.rstrip("/") + "/",
                delimiter="/",
                max_keys=1,
                headers=self.additional_header,
            )
            if out.object_list or out.prefix_list:
                return {
                    "Key": "/".join([bucket_name, key]),
                    "name": "/".join([bucket_name, key]),
                    "type": "directory",
                    "Size": 0,
                    "size": 0,
                    "StorageClass": "DIRECTORY",
                }
            raise FileNotFoundError(path)
        except exceptions.NoSuchBucket:
            raise FileNotFoundError(path)

    def info(self, path, version_id=None, refresh=False):
        path = self._strip_protocol(path)
        if path in ["/", ""]:
            return {"name": path, "size": 0, "type": "directory"}
        kwargs = self.kwargs.copy()
        if version_id is not None:
            if not self.version_aware:
                raise ValueError(
                    "version_id cannot be specified if the "
                    "filesystem is not version aware"
                )
        bucket_name, key, path_version_id = self.split_path(path)
        version_id = _coalesce_version_id(path_version_id, version_id)

        should_fetch_from_oss = (key and self._ls_from_cache(path) is None) or refresh

        if should_fetch_from_oss:
            return self._info(path, bucket_name, key, kwargs, version_id)
        return super().info(path)

    def checksum(self, path, refresh=False):
        info = self.info(path, refresh=refresh)

        if info["type"] != "directory":
            return int(info["ETag"].strip('"').split("-")[0], 16)
        else:
            return int(tokenize(info), 16)

    def _lsdir(self, path, refresh=False, max_items=100, delimiter="/"):
        bucket_name, prefix, _ = self.split_path(path)
        prefix = prefix + "/" if prefix else ""
        if path not in self.dircache or refresh or not delimiter:
            try:
                logger.debug("Get directory listing page for %s" % path)
                bucket = self._make_bucket(bucket_name)
                files = []
                dircache = []
                for obj in oss2.ObjectIteratorV2(
                    bucket,
                    prefix=prefix,
                    delimiter=delimiter,
                    max_keys=max_items,
                    max_retries=self.retries,
                ):
                    if obj.is_prefix():
                        dircache.append(obj.key)
                    else:
                        files.append(
                            {
                                "Key": obj.key,
                                "Size": obj.size,
                                "StorageClass": "STANDARD",
                                "type": "file",
                                "size": obj.size,
                                "LastModified": datetime.fromtimestamp(
                                    obj.last_modified
                                ),
                                "ETag": obj.etag,
                            }
                        )
                files.extend(
                    [
                        {
                            "Key": d[:-1],
                            "Size": 0,
                            "StorageClass": "DIRECTORY",
                            "type": "directory",
                            "size": 0,
                        }
                        for d in dircache
                    ]
                )
                for f in files:
                    f["Key"] = "/".join([bucket_name, f["Key"]])
                    f["name"] = f["Key"]
            except exceptions.NoSuchBucket:
                raise FileNotFoundError(bucket_name)

            if delimiter:
                self.dircache[path] = files
            return files
        return self.dircache[path]

    def _lsbuckets(self, refresh=False):
        if "" not in self.dircache or refresh:
            if self.anon:
                # cannot list buckets if not logged in
                return []
            try:
                service = self._make_service()
                bucket_names = [b.name for b in oss2.BucketIterator(service)]
            except Exception:
                # list bucket permission missing
                return []
            files = []
            for name in bucket_names:
                files.append(
                    {
                        "Key": name,
                        "Size": 0,
                        "StorageClass": "BUCKET",
                        "size": 0,
                        "type": "directory",
                        "name": name,
                    }
                )
            self.dircache[""] = files
            return files
        return self.dircache[""]

    def _ls(self, path, refresh=False):
        """List files in given bucket, or list of buckets.

        Listing is cached unless `refresh=True`.

        Note: only your buckets associated with the login will be listed by
        `ls('')`, not any public buckets (even if already accessed).

        Parameters
        ----------
        path : string/bytes
            location at which to list files
        refresh : bool (=False)
            if False, look in local cache for file details first
        """
        path = self._strip_protocol(path)
        if path in ["", "/"]:
            return self._lsbuckets(refresh)
        else:
            return self._lsdir(path, refresh)

    def ls(self, path, detail=False, refresh=False, **kwargs):
        path = self._strip_protocol(path).rstrip("/")
        files = self._ls(path, refresh=refresh)
        if not files:
            files = self._ls(self._parent(path), refresh=refresh)
            files = [
                o
                for o in files
                if o["name"].rstrip("/") == path and o["type"] != "directory"
            ]
        if detail:
            return files
        else:
            return list(sorted(set([f["name"] for f in files])))

    def isdir(self, path):
        path = self._strip_protocol(path).strip("/")
        # Send buckets to super
        if "/" not in path:
            return super(OSSFileSystem, self).isdir(path)

        if path in self.dircache:
            for fp in self.dircache[path]:
                # For files the dircache can contain itself.
                # If it contains anything other than itself it is a directory.
                if fp["name"] != path:
                    return True
            return False

        parent = self._parent(path)
        if parent in self.dircache:
            for f in self.dircache[parent]:
                if f["name"] == path:
                    # If we find ourselves return whether we are a directory
                    return f["type"] == "directory"
            return False

        # This only returns things within the path and NOT the path object itself
        return bool(self._lsdir(path))

    def object_version_info(self, path, **kwargs):
        if not self.version_aware:
            raise ValueError(
                "version specific functionality is disabled for "
                "non-version aware filesystems"
            )
        bucket_name, key, _ = self.split_path(path)
        versions = []
        is_truncated = True
        while is_truncated:
            out = self.call_oss(
                bucket_name,
                "list_object_versions",
                key,
                headers=self.additional_header,
                **kwargs
            )
            versions.extend(out.versions)
            kwargs.update(
                {
                    "versionid_marker": out.next_versionid_marker,
                    "key_marker": out.next_key_marker,
                }
            )
            is_truncated = out.is_truncated
        return versions

    _metadata_cache = {}

    def metadata(self, path, refresh=False, **kwargs):
        """Return metadata of path.

        Metadata is cached unless `refresh=True`.

        Parameters
        ----------
        path : string/bytes
            filename to get metadata for
        refresh : bool (=False)
            if False, look in local cache for file metadata first
        """
        bucket_name, key, version_id = self.split_path(path)
        if refresh or path not in self._metadata_cache:
            kwargs.update(self.additional_header)
            response = self.call_oss(
                bucket_name,
                "head_object",
                key,
                headers=kwargs,
                params=version_id_kw(version_id),
            )
            meta = response.headers
            self._metadata_cache[path] = meta

        return self._metadata_cache[path]

    def get_tags(self, path: str) -> dict:
        """Retrieve tag key/values for the given path

        Returns
        -------
        {str: str}
        """
        bucket_name, key, version_id = self.split_path(path)
        response = self.call_oss(
            bucket_name, "get_object_tagging", key, params=version_id_kw(version_id)
        )
        return response.tag_set.tagging_rule

    def put_tags(self, path: str, tags: dict, mode="o"):
        """Set tags for given existing key

        Tags are a str:str mapping that can be attached to any key, see
        https://help.aliyun.com/document_detail/121939.html?spm=a2c4g.11186623.6.1074.5eca231dYheL2S
        This is similar to, but distinct from, key metadata, which is usually
        set at key creation time.

        Parameters
        ----------
        path: str
            Existing key to attach tags to
        tags: dict str, str
            Tags to apply.
        mode:
            One of 'o' or 'm'
            'o': Will over-write any existing tags.
            'm': Will merge in new tags with existing tags.  Incurs two remote
            calls.
        """
        bucket_name, key, version_id = self.split_path(path)

        rule = oss2.models.TaggingRule()
        if mode == "m":
            existing_tags = self.get_tags(path=path)
            tags.update(existing_tags)
        elif mode == "o":
            pass
        else:
            raise ValueError("Mode must be {'o', 'm'}, not %s" % mode)
        for k, v in tags.items():
            rule.add(k, v)
        self.call_oss(
            bucket_name,
            "put_object_tagging",
            key,
            oss2.models.Tagging(rule),
            headers=self.additional_header,
            params=version_id_kw(version_id),
        )

    def getxattr(self, path, attr_name, **kwargs):
        """Get an attribute from the metadata.

        Examples
        --------
        >>> myossfs.getxattr('mykey', 'attribute_1')  # doctest: +SKIP
        'value_1'
        """
        attr_name = attr_name.replace("_", "-")
        xattr = self.metadata(path, **kwargs)
        if attr_name in xattr:
            return xattr[attr_name]
        return None

    def setxattr(self, path, copy_kwargs=None, **kw_args):
        """Set metadata.

        Attributes have to be of the form documented in the
        `Metadata Reference`_.

        Parameters
        ----------
        kw_args : key-value pairs like field="value", where the values must be
            strings. Does not alter existing fields, unless
            the field appears here - if the value is None, delete the
            field.
        copy_kwargs : dict, optional
            dictionary of additional params to use for the underlying
            s3.copy_object.

        Examples
        --------
        >>> myossfile.setxattr(attribute_1='value1', attribute_2='value2')

        # Example for use with copy_args
        >>> myossfile.setxattr(copy_kwargs={'ContentType': 'application/pdf'},
                               attribute_1='value1')
        """

        kw_args = {k.replace("_", "-"): v for k, v in kw_args.items()}
        bucket_name, key, version_id = self.split_path(path)
        metadata = self.metadata(path)
        metadata.update(**kw_args)
        copy_kwargs = copy_kwargs or {}

        # remove all keys that are None
        for kw_key in kw_args:
            if kw_args[kw_key] is None:
                metadata.pop(kw_key, None)
        self.call_oss(bucket_name, "update_object_meta", key, headers=metadata)

        # refresh metadata
        self._metadata_cache[path] = metadata

    def chmod(self, path, acl, **kwargs):
        """Set Access Control on a bucket/key

        See https://help.aliyun.com/document_detail/145658.html?spm=a2c4g.11186623.6.1022.52a110ad0rckwF  # noqa: E501
            https://help.aliyun.com/document_detail/88455.html?spm=a2c4g.11186623.6.1052.32ad48f2yChAqb  # noqa: E501

        Parameters
        ----------
        path : string
            the object to set
        acl : string
            the value of ACL to apply
        """
        bucket_name, key, version_id = self.split_path(path)
        self.call_oss(
            bucket_name,
            "put_object_acl",
            key,
            key_acls[acl],
            params=version_id_kw(version_id),
            headers=kwargs,
        )
        if key:
            if acl not in key_acls:
                raise ValueError("ACL not in %s", key_acls)
        else:
            if acl not in buck_acls:
                raise ValueError("ACL not in %s", buck_acls)
            self.call_oss(bucket_name, "put_bucket_acl", buck_acls[acl])

    def url(self, path, expires=3600, **kwargs):
        """Generate presigned URL to access path by HTTP

        Parameters
        ----------
        path : string
            the key path we are interested in
        expires : int
            the number of seconds this signature will be good for.
        """
        bucket_name, key, version_id = self.split_path(path)
        return self.call_oss(
            bucket_name,
            "sign_url",
            "GET",
            key,
            expires,
            headers=kwargs,
            params=version_id_kw(version_id),
        )

    def merge(self, path, filelist, **kwargs):
        """Create single OSS file from list of OSS files

        Uses multi-part, no data is downloaded. The original files are
        not deleted.

        Parameters
        ----------
        path : str
            The final file to produce
        filelist : list of str
            The paths, in order, to assemble into the final file.
        """
        bucket_name, key, version_id = self.split_path(path)
        if version_id:
            raise ValueError("Cannot write to an explicit versioned file!")
        # TODO: support versions
        mpu = self.call_oss(bucket_name, "init_multipart_upload", key, headers=kwargs)

        out = [
            self.call_oss(
                bucket_name,
                "upload_part_copy",
                self.split_path(f)[0],
                self.split_path(f)[1],
                None,
                key,
                mpu.upload_id,
                i + 1,
            )
            for i, f in enumerate(filelist)
        ]

        parts = [PartInfo(i + 1, o.etag) for i, o in enumerate(out)]
        self.call_oss(
            bucket_name, "complete_multipart_upload", key, mpu.upload_id, parts
        )

        self.invalidate_cache(path)

    def _copy_basic(self, path1, path2, **kwargs):
        """Copy file between locations on OSS

        Not allowed where the origin is >5GB - use copy_managed
        """
        buc1, key1, ver1 = self.split_path(path1)
        buc2, key2, ver2 = self.split_path(path2)
        if ver2:
            raise ValueError("Cannot copy to a versioned file!")
        try:
            self.call_oss(
                buc2, "copy_object", buc1, key1, key2, params=version_id_kw(ver1)
            )
        except Exception as e:
            raise ValueError("Copy failed (%r -> %r): %s" % (path1, path2, e)) from e
        self.invalidate_cache(path2)

    def _copy_managed(self, path1, path2, size, block=5 * 2**30, **kwargs):
        """Copy file between locations on OSS as multi-part

        block: int
            The size of the pieces, must be larger than 5MB and at most 5GB.
            Smaller blocks mean more calls, only useful for testing.
        """
        if block < 5 * 2**20 or block > 5 * 2**30:
            raise ValueError("Copy block size must be 5MB<=block<=5GB")
        buc1, key1, ver1 = self.split_path(path1)
        buc2, key2, ver2 = self.split_path(path2)
        if ver2:
            raise ValueError("Cannot copy to a versioned file!")
        mpu = self.call_oss(buc2, "init_multipart_upload", key2, headers=kwargs)

        out = [
            self.call_oss(
                buc2,
                "upload_part_copy",
                buc1,
                key1,
                brange,
                key2,
                mpu.upload_id,
                i + 1,
                params=version_id_kw(ver1),
            )
            for i, brange in enumerate(_get_brange(size, block))
        ]

        parts = [PartInfo(i + 1, o.etag) for i, o in enumerate(out)]
        self.call_oss(buc2, "complete_multipart_upload", key2, mpu.upload_id, parts)
        self.invalidate_cache(path2)

    def cp_file(self, path1, path2, **kwargs):
        gb5 = 5 * 2**30
        path1 = self._strip_protocol(path1)
        size = self._info(path1)["size"]
        if size <= gb5:
            # simple copy allowed for <5GB
            self._copy_basic(path1, path2, **kwargs)
        else:
            # serial multipart copy
            self._copy_managed(path1, path2, size, **kwargs)

    def _clear_multipart_uploads(self, bucket_name):
        """Remove any partial uploads in the bucket"""
        out = self.call_oss(bucket_name, "list_multipart_uploads")
        for upload in out.upload_list:
            self.call_oss(
                bucket_name, "abort_multipart_upload", upload.key, upload.upload_id
            )

    def _bulk_delete(self, pathlist, **kwargs):
        if not pathlist:
            return
        bucket_names = {self.split_path(path)[0] for path in pathlist}
        if len(bucket_names) > 1:
            raise ValueError("Bulk delete files should refer to only one bucket")
        bucket_name = bucket_names.pop()
        if len(pathlist) > 1000:
            raise ValueError("Max number of files to delete in one call is 1000")
        key_list = [self.split_path(path)[1] for path in pathlist]

        for path in pathlist:
            self.invalidate_cache(self._parent(path))
        try:
            self.call_oss(bucket_name, "batch_delete_objects", key_list)
        except exceptions.NoSuchBucket:
            raise FileNotFoundError(",".join(key_list))

    def _rm(self, paths, **kwargs):
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        files = [p for p in paths if self.split_path(p)[1]]
        dirs = [p for p in paths if not self.split_path(p)[1]]
        # TODO: fails if more than one bucket in list
        for i in range(0, len(files), 1000):
            self._bulk_delete(files[i : i + 1000])
        for d in dirs:
            self.rmdir(d)
        for p in paths:
            self.invalidate_cache(p)
            self.invalidate_cache(self._parent(p))

    def rm(self, path, recursive=False, **kwargs):
        if recursive and isinstance(path, str):
            bucket_name, key, _ = self.split_path(path)
            if not key and self.is_bucket_versioned(bucket_name):
                # special path to completely remove versioned bucket
                self._rm_versioned_bucket_contents(bucket_name)
        super().rm(path, recursive=recursive, **kwargs)

    def is_bucket_versioned(self, bucket_name):
        return self.call_oss(bucket_name, "get_bucket_versioning").status == "Enabled"

    def _rm_versioned_bucket_contents(self, bucket_name):
        """Remove a versioned bucket and all contents"""
        next_key_marker = None
        next_versionid_marker = None
        while True:
            result = self.call_oss(
                bucket_name,
                "list_object_versions",
                key_marker=next_key_marker,
                versionid_marker=next_versionid_marker,
            )
            version_list = oss2.models.BatchDeleteObjectVersionList()
            for o in itertools.chain(result.versions, result.delete_marker):
                version_list.append(
                    oss2.models.BatchDeleteObjectVersion(o.key, o.versionid)
                )
            if version_list.len() == 0:
                break
            self.call_oss(bucket_name, "delete_object_versions", version_list)

            is_truncated = result.is_truncated

            # If list results is not complete, continue to list from marker
            if is_truncated:
                next_key_marker = result.next_key_marker
                next_versionid_marker = result.next_versionid_marker
            else:
                break

    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
        else:
            path = self._strip_protocol(path)
            self.dircache.pop(path, None)
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

    def walk(self, path, maxdepth=None, **kwargs):
        if path in ["", "*"] + ["{}://".format(p) for p in self.protocol]:
            raise ValueError("Cannot crawl all of OSS")
        return super().walk(path, maxdepth=maxdepth, **kwargs)

    def modified(self, path, refresh=False):
        """Return the last modified timestamp of file at `path` as a datetime"""
        info = self.info(path=path, refresh=refresh)
        if "LastModified" not in info:
            # This path is a bucket or folder, which do not currently
            # have a modified date
            raise IsADirectoryError
        return info["LastModified"].replace(tzinfo=None)

    def sign(self, path, expiration=100, **kwargs):
        return self.url(path, expiration, **kwargs)

    def created(self, path):
        pass


class OSSFile(AbstractBufferedFile):
    retries = 5
    part_min = 5 * 2**20
    part_max = 5 * 2**30

    def __init__(
        self,
        oss,
        path,
        mode="rb",
        block_size=5 * 2**20,
        version_id=None,
        fill_cache=True,
        autocommit=True,
        cache_type="bytes",
    ):
        bucket_name, key, path_version_id = oss.split_path(path)
        if not key:
            raise ValueError("Attempt to open non key-like path: %s" % path)
        self.bucket_name = bucket_name
        self.key = key
        self.version_id = _coalesce_version_id(version_id, path_version_id)

        self.mpu = None
        self.parts = None
        self.fill_cache = fill_cache

        if "r" not in mode:
            if block_size < 5 * 2**20:
                raise ValueError("Block size must be >=5MB")
        else:
            if version_id and oss.version_aware:
                self.version_id = version_id
                self.details = oss.info(path, version_id=version_id)
                self.size = self.details["size"]
            elif oss.version_aware:
                # In this case we have not managed to get the VersionId out of
                # details and we should invalidate the cache and perform a full
                # head_object since it has likely been partially populated by ls.
                oss.invalidate_cache(path)
                self.details = oss.info(path)
                self.version_id = self.details.get("VersionId")
        super().__init__(
            oss, path, mode, block_size, autocommit=autocommit, cache_type=cache_type
        )
        self.oss = self.fs  # compatibility

        # when not using autocommit we want to have transactional state to manage
        self.append_block = False

        if "a" in mode and oss.exists(path):
            loc = oss.info(path)["size"]
            if loc < 5 * 2**20:
                # existing file too small for multi-upload: download
                self.write(self.fs.cat(self.path))
            else:
                self.append_block = True
            self.loc = loc

    def _call_oss(self, method, *args, **kwargs):
        return self.fs.call_oss(self.bucket_name, method, *args, **kwargs)

    def _initiate_upload(self):
        if self.autocommit and not self.append_block and self.tell() < self.blocksize:
            # only happens when closing small file, use on-shot PUT
            return
        logger.debug("Initiate upload for %s" % self)
        self.parts = []
        self.mpu = self._call_oss("init_multipart_upload", self.key)

        if self.append_block:
            # use existing data in key when appending,
            # and block is big enough
            result = self._call_oss(
                "upload_part_copy",
                self.bucket_name,
                self.key,
                byte_range=None,
                target_key=self.key,
                target_upload_id=self.mpu.upload_id,
                target_part_number=1,
            )
            self.parts.append(PartInfo(1, result.etag))

    def metadata(self, refresh=False, **kwargs):
        """Return metadata of file.
        See :func:`~s3fs.S3Filesystem.metadata`.

        Metadata is cached unless `refresh=True`.
        """
        return self.fs.metadata(self.path, refresh, **kwargs)

    def getxattr(self, xattr_name, **kwargs):
        """Get an attribute from the metadata.
        See :func:`~s3fs.S3Filesystem.getxattr`.

        Examples
        --------
        >>> mys3file.getxattr('attribute_1')  # doctest: +SKIP
        'value_1'
        """
        return self.fs.getxattr(self.path, xattr_name, **kwargs)

    def setxattr(self, copy_kwargs=None, **kwargs):
        """Set metadata.
        See :func:`~s3fs.S3Filesystem.setxattr`.

        Examples
        --------
        >>> mys3file.setxattr(attribute_1='value1', attribute_2='value2')
        """
        if self.writable():
            raise NotImplementedError(
                "cannot update metadata while file " "is open for writing"
            )
        return self.fs.setxattr(self.path, copy_kwargs=copy_kwargs, **kwargs)

    def url(self, **kwargs):
        """HTTP URL to read this file (if it already exists)"""
        return self.fs.url(self.path, **kwargs)

    def _fetch_range(self, start, end):
        return _fetch_range(
            self.fs, self.bucket_name, self.key, self.version_id, start, end
        )

    def _upload_chunk(self, final=False):
        logger.debug(
            "Upload for %s, final=%s, loc=%s, buffer loc=%s"
            % (self, final, self.loc, self.buffer.tell())
        )
        if (
            self.autocommit
            and not self.append_block
            and final
            and self.tell() < self.blocksize
        ):
            # only happens when closing small file, use on-shot PUT
            data1 = False
        else:
            self.buffer.seek(0)
            (data0, data1) = (None, self.buffer.read(self.blocksize))

        while data1:
            (data0, data1) = (data1, self.buffer.read(self.blocksize))
            data1_size = len(data1)

            if 0 < data1_size < self.blocksize:
                remainder = data0 + data1
                remainder_size = self.blocksize + data1_size

                if remainder_size <= self.part_max:
                    (data0, data1) = (remainder, None)
                else:
                    partition = remainder_size // 2
                    (data0, data1) = (remainder[:partition], remainder[partition:])

            part = len(self.parts) + 1
            logger.debug("Upload chunk %s, %s" % (self, part))
            result = self._call_oss(
                "upload_part", self.key, self.mpu.upload_id, part, data0
            )
            self.parts.append(PartInfo(part, result.etag))

        if self.autocommit and final:
            self.commit()
        return not final

    def commit(self):
        logger.debug("Commit %s" % self)
        if self.tell() == 0:
            if self.buffer is not None:
                logger.debug("Empty file committed %s" % self)
                self._abort_mpu()
                result = self.fs.touch(self.path)
        elif not self.parts:
            if self.buffer is not None:
                logger.debug("One-shot upload of %s" % self)
                self.buffer.seek(0)
                data = self.buffer.read()
                result = self._call_oss("put_object", self.key, data)
            else:
                raise RuntimeError
        else:
            logger.debug("Complete multi-part upload for %s " % self)
            result = self._call_oss(
                "complete_multipart_upload", self.key, self.mpu.upload_id, self.parts
            )
        if self.fs.version_aware:
            self.version_id = result.versionid

        # complex cache invalidation, since file's appearance can cause several
        # directories
        self.buffer = None
        parts = self.path.split("/")
        path = parts[0]
        for p in parts[1:]:
            if path in self.fs.dircache and not [
                True for f in self.fs.dircache[path] if f["name"] == path + "/" + p
            ]:
                self.fs.invalidate_cache(path)
            path = path + "/" + p

    def _abort_mpu(self):
        if self.mpu:
            self._call_oss("abort_multipart_upload", self.key, self.mpu.upload_id)
            self.mpu = None

    def discard(self):
        self._abort_mpu()
        self.buffer = None  # file becomes unusable


def _fetch_range(fs, bucket_name, key, version_id, start, end, params=None):
    if params is None:
        params = {}
    params.update(version_id_kw(version_id))
    if start == end:
        logger.debug(
            "skip fetch for negative range - bucket=%s,key=%s,start=%d,end=%d",
            bucket_name,
            key,
            start,
            end,
        )
        return b""
    logger.info("Fetch: %s/%s, %s-%s", bucket_name, key, start, end)

    resp = fs.call_oss(
        bucket_name, "get_object", key, byte_range=(start, end - 1), params=params
    )
    data = resp.read()
    resp.close()
    return data
