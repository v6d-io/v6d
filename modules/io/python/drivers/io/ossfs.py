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

import asyncio
import logging
import os
import socket
from typing import Optional, Tuple

import oss2
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options, tokenize
from oss2 import exceptions, headers
from oss2.models import PartInfo

logger = logging.getLogger("vineyard.io.ossfs")


def setup_logging(level=None):
    handle = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s " "- %(message)s")
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(level or os.environ["OSSFS_LOGGING_LEVEL"])


_VALID_FILE_MODES = {"r", "w", "a", "rb", "wb", "ab"}


class OSSFileSystem(AbstractFileSystem):
    root_marker = ""
    connect_timeout = 5
    retries = 5
    read_timeout = 15
    default_block_size = 5 * 2**20
    protocol = ["oss"]
    _extra_tokenize_attributes = ("default_block_size", )

    def __init__(self,
                 anon=False,
                 key=None,
                 secret=None,
                 token=None,
                 endpoint=None,
                 use_ssl=True,
                 client_kwargs=None,
                 requester_pays=False,
                 default_block_size=None,
                 default_fill_cache=True,
                 default_cache_type="bytes",
                 version_aware=False,
                 config_kwargs=None,
                 oss_additional_kwargs=None,
                 session=None,
                 username=None,
                 password=None,
                 asynchronous=False,
                 loop=None,
                 **kwargs):
        if key and username:
            raise KeyError("Supply either key or username, not both")
        if secret and password:
            raise KeyError("Supply secret or password, not both")
        if username:
            key = username
        if password:
            secret = password

        self.anon = anon
        self.key = key
        self.secret = secret
        self.token = token
        self.endpoint = endpoint
        self.kwargs = kwargs
        super_kwargs = {
            k: kwargs.pop(k)
            for k in ["use_listings_cache", "listings_expiry_time", "max_paths"] if k in kwargs
        }  # passed to fsspec superclass
        super().__init__(loop=loop, asynchronous=asynchronous, **super_kwargs)

        self.default_block_size = default_block_size or self.default_block_size
        self.default_fill_cache = default_fill_cache
        self.default_cache_type = default_cache_type
        self.version_aware = version_aware
        self.client_kwargs = client_kwargs or {}
        self.config_kwargs = config_kwargs or {}
        self.req_kw = {headers.OSS_REQUEST_PAYER: "requester"} if requester_pays else {}
        self.oss_additional_kwargs = oss_additional_kwargs or {}
        self.auth = oss2.Auth(key, secret)

    def split_path(self, path) -> Tuple[str, str]:
        path = self._strip_protocol(path)
        path = path.lstrip("/")
        if "/" not in path:
            return path, ""
        else:
            return path.split("/", 1)

    def _open(self,
              path,
              mode="rb",
              endpoint="http://oss-cn-hangzhou.aliyuncs.com",
              block_size=None,
              acl="",
              version_id=None,
              fill_cache=None,
              cache_type=None,
              autocommit=True,
              requester_pays=None,
              **kwargs):
        if block_size is None:
            block_size = self.default_block_size
        if fill_cache is None:
            fill_cache = self.default_fill_cache
        if requester_pays is None:
            requester_pays = bool(self.req_kw)

        acl = acl or self.oss_additional_kwargs.get("ACL", "")
        kw = self.oss_additional_kwargs.copy()
        kw.update(kwargs)

        if cache_type is None:
            cache_type = self.default_cache_type

        return OSSFile(
            self,
            path,
            mode,
            endpoint,
            block_size=block_size,
            acl=acl,
            version_id=version_id,
            fill_cache=fill_cache,
            oss_additional_kwargs=kw,
            cache_type=cache_type,
            autocommit=autocommit,
            requester_pays=requester_pays,
        )

    def exists(self, path, **kwargs):
        if path in ["", "/"]:
            # the root always exists, even if anon
            return True
        bucket, key = self.split_path(path)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)

        try:
            bucket.get_bucket_info()
        except oss2.exceptions.NoSuchBucket:
            return False
        if key:
            return bucket.object_exists(key)
        else:
            return True

    def _cat_file(self, path, version_id=None, start=None, end=None):
        bucket, key = self.split_path(path)
        if (start is None) ^ (end is None):
            raise ValueError("Give start and end or neither")
        if start is not None and end is not None:
            byte_range = (start, end)
        else:
            byte_range = None
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        stream = bucket.get_object(key, byte_range)
        data = stream.read()
        stream.close()
        return data

    def _pipe_file(self, path, data, chunksize=50 * 2**20, **kwargs):
        bucket, key = self.split_path(path)
        size = len(data)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        if size < 5 * 2**20:
            return bucket.put_object(key, data)
        else:
            mpu = bucket.init_multipart_upload(key, **kwargs)

            out = [
                bucket.upload_part(key, mpu.upload_id, i + 1, data[off:off + chunksize])
                for i, off in enumerate(range(0, len(data), chunksize))
            ]

            parts = [PartInfo(i + 1, o.etag) for i, o in enumerate(out)]
            bucket.complete_multipart_upload(key, mpu.upload_id, parts)
        self.invalidate_cache(path)

    def _put_file(self, lpath, rpath, **kwargs):
        bucket, key = self.split_path(rpath)
        if os.path.isdir(lpath) and key:
            # don't make remote "directory"
            return
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        bucket.put_object_from_file(key, lpath)
        self.invalidate_cache(rpath)

    def _get_file(self, rpath, lpath):
        bucket, key = self.split_path(rpath)
        if os.path.isdir(lpath):
            return
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        bucket.get_object_to_file(key, rpath)

    def _info(self, path, kwargs={}, version_id=None):
        bucket, key = self.split_path(path)
        try:
            bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
            out = bucket.head_object(key)
            return {
                "ETag": out.etag,
                "Key": path,
                "LastModified": out.last_modified,
                "Size": out.content_length,
                "size": out.content_length,
                "name": path,
                "type": "file",
                "StorageClass": "STANDARD",
                "VersionId": out.versionid,
            }
        except exceptions.NotFound:
            pass
        except Exception as e:
            raise e

        try:
            # We check to see if the path is a directory by attempting to list its
            # contexts. If anything is found, it is indeed a directory
            out = bucket.list_objects_v2(key.rstrip("/") + "/", delimiter="/", max_keys=1)
            if out.object_list:
                return {
                    "Key": path,
                    "name": path,
                    "type": "directory",
                    "Size": 0,
                    "size": 0,
                    "StorageClass": "DIRECTORY",
                }

            raise FileNotFoundError(path)
        except Exception as e:
            raise e

    def info(self, path, version_id=None, refresh=False):
        path = self._strip_protocol(path)
        if path in ["/", ""]:
            return {"name": path, "size": 0, "type": "directory"}
        kwargs = self.kwargs.copy()

        bucket, key = self.split_path(path)
        should_fetch_from_oss = (key and self._ls_from_cache(path) is None) or refresh

        if should_fetch_from_oss:
            return self._info(path, kwargs, version_id)
        return super().info(path)

    def checksum(self, path, refresh=False):
        info = self.info(path, refresh=refresh)

        if info["type"] != "directory":
            return int(info["ETag"].strip('"').split("-")[0], 16)
        else:
            return int(tokenize(info), 16)

    def _lsdir(self, path, refresh=False, max_items=None, delimiter="/"):
        bucket, prefix = self.split_path(path)
        prefix = prefix + "/" if prefix else ""
        if path not in self.dircache or refresh or not delimiter:
            try:
                logger.debug("Get directory listing page for %s" % path)
                bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
                if max_items is None:
                    max_items = 100
                out = bucket.list_objects_v2(prefix, delimiter, max_keys=max_items)
                files = []
                dircache = []
                dircache.extend(out.prefix_list)
                files.extend([{
                    "Key": o.key,
                    "Size": o.size,
                    "StorageClass": "STANDARD",
                    "type": "file",
                    "size": o.size,
                } for o in out.object_list])
                files.extend([{
                    "Key": d,
                    "Size": 0,
                    "StorageClass": "DIRECTORY",
                    "type": "directory",
                    "size": 0,
                } for d in dircache])
                for f in files:
                    f["Key"] = "/".join([bucket.bucket_name, f["Key"]])
                    f["name"] = f["Key"]
            except Exception as e:
                raise e

            if delimiter:
                self.dircache[path] = files
            return files
        return self.dircache[path]

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

    def _lsbuckets(self, refresh=False):
        if "" not in self.dircache or refresh:
            if self.anon:
                # cannot list buckets if not logged in
                return []
            try:
                service = oss2.Service(auth, self.endpoint)
                files = [b.name for b in oss2.BucketIterator(service)]
            except Exception as e:
                # list bucket permission missing
                raise e
            for f in files:
                f["Key"] = f["Name"]
                f["Size"] = 0
                f["StorageClass"] = "BUCKET"
                f["size"] = 0
                f["type"] = "directory"
                f["name"] = f["Name"]
                del f["Name"]
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
            files = [o for o in files if o["name"].rstrip("/") == path and o["type"] != "directory"]
        if detail:
            return files
        else:
            return list(sorted(set([f["name"] for f in files])))

    def touch(self, path, truncate=True, data=None, **kwargs):
        """Create empty file or truncate"""
        bucket, key = self.split_path(path)
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        if not truncate and self.exists(path):
            raise ValueError("S3 does not support touching existent files")
        try:
            write_result = bucket.put_object(key, data="")
        except Exception as ex:
            raise ex
        self.invalidate_cache(self._parent(path))
        return write_result

    def _clear_multipart_uploads(self, bucket):
        """Remove any partial uploads in the bucket"""
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        out = bucket.list_multipart_uploads()
        for upload in out.upload_list:
            bucket.abort_multipart_upload(upload.key, upload.upload_id)

    def _bulk_delete(self, pathlist, **kwargs):
        if not pathlist:
            return
        buckets = {self.split_path(path)[0] for path in pathlist}
        if len(buckets) > 1:
            raise ValueError("Bulk delete files should refer to only one bucket")
        bucket = buckets.pop()
        if len(pathlist) > 1000:
            raise ValueError("Max number of files to delete in one call is 1000")
        key_list = [self.split_path(path)[1] for path in pathlist]

        for path in pathlist:
            self.invalidate_cache(self._parent(path))
        bucket = oss2.Bucket(self.auth, self.endpoint, bucket)
        bucket.batch_delete_objects(key_list)

    def _rm(self, paths, **kwargs):
        files = [p for p in paths if self.split_path(p)[1]]
        dirs = [p for p in paths if not self.split_path(p)[1]]
        # TODO: fails if more than one bucket in list
        for i in range(0, len(files), 1000):
            self._bulk_delete(files[i:i + 1000])
        for p in paths:
            self.invalidate_cache(p)
            self.invalidate_cache(self._parent(p))

    def rm(self, path, recursive=False, **kwargs):
        super().rm(path, recursive=recursive, **kwargs)

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

    def modified(self, path, version_id=None, refresh=False):
        """Return the last modified timestamp of file at `path` as a datetime"""
        info = self.info(path=path, version_id=version_id, refresh=refresh)
        if "LastModified" not in info:
            # This path is a bucket or folder, which do not currently have a modified date
            raise IsADirectoryError
        return info["LastModified"]


class OSSFile(AbstractBufferedFile):
    retries = 5
    part_min = 5 * 2**20
    part_max = 5 * 2**30

    def __init__(
        self,
        oss,
        path,
        mode="rb",
        endpoint="http://oss-cn-hangzhou.aliyuncs.com",
        block_size=5 * 2**20,
        acl="",
        version_id=None,
        fill_cache=True,
        oss_additional_kwargs=None,
        autocommit=True,
        cache_type="bytes",
        requester_pays=False,
    ):
        bucket_name, key = oss.split_path(path)
        if not key:
            raise ValueError("Attempt to open non key-like path: %s" % path)
        self.bucket_name = bucket_name
        self.key = key
        self.acl = acl

        self.bucket = oss2.Bucket(oss.auth, oss.endpoint, self.bucket_name)

        self.mpu = None
        self.parts = None
        self.fill_cache = fill_cache
        self.oss_additional_kwargs = oss_additional_kwargs or {}
        self.req_kw = {"RequestPayer": "requester"} if requester_pays else {}

        super().__init__(oss, path, mode, block_size, autocommit=autocommit, cache_type=cache_type)
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

    def _initiate_upload(self):
        if self.autocommit and not self.append_block and self.tell() < self.blocksize:
            # only happens when closing small file, use on-shot PUT
            return
        logger.debug("Initiate upload for %s" % self)
        self.parts = []
        self.mpu = self.bucket.init_multipart_upload(self.key)

        if self.append_block:
            # use existing data in key when appending,
            # and block is big enough
            result = self.bucket.upload_part_copy(
                self.bucket_name,
                self.key,
                byte_range=None,
                target_key=self.key,
                target_upload_id=self.mpu.upload_id,
                target_part_number=1,
            )
            self.parts.append(PartInfo(1, result.etag))

    def _fetch_range(self, start, end):
        if start == end:
            logger.debug(
                "skip fetch for negative range - bucket=%s,key=%s,start=%d,end=%d",
                self.bucket_name,
                self.key,
                start,
                end,
            )
            return b""
        logger.info("Fetch: %s/%s, %s-%s", self.bucket_name, self.key, start, end)

        resp = self.bucket.get_object(self.key, byte_range=(start, end - 1))
        data = resp.read()
        resp.close()
        return data

    def _upload_chunk(self, final=False):
        bucket, key = self.fs.split_path(self.path)
        logger.debug("Upload for %s, final=%s, loc=%s, buffer loc=%s" % (self, final, self.loc, self.buffer.tell()))
        if (self.autocommit and not self.append_block and final and self.tell() < self.blocksize):
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

            result = self.bucket.upload_part(self.key, self.mpu.upload_id, part, data0)

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
                write_result = self.fs.touch(self.path)
        elif not self.parts:
            if self.buffer is not None:
                logger.debug("One-shot upload of %s" % self)
                self.buffer.seek(0)
                data = self.buffer.read()
                write_result = self.bucket.put_object(self.key, data)
            else:
                raise RuntimeError
        else:
            logger.debug("Complete multi-part upload for %s " % self)
            write_result = self.bucket.complete_multipart_upload(self.key, self.mpu.upload_id, self.parts)

        # complex cache invalidation, since file's appearance can cause several
        # directories
        self.buffer = None
        parts = self.path.split("/")
        path = parts[0]
        for p in parts[1:]:
            if path in self.fs.dircache and not [True for f in self.fs.dircache[path] if f["name"] == path + "/" + p]:
                self.fs.invalidate_cache(path)
            path = path + "/" + p

    def _abort_mpu(self):
        if self.mpu:
            self.bucket.abort_multipart_upload(self.key, self.mpu.upload_id)
            self.mpu = None

    def discard(self):
        self._abort_mpu()
        self.buffer = None  # file becomes unusable
