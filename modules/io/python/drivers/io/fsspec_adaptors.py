#! /usr/bin/env python3
# -*- coding: utf-8 -*-
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

import fsspec
import fsspec.implementations.arrow

# register OSS
try:
    from vineyard.drivers.io import ossfs
except ImportError:
    ossfs = None

if ossfs:
    fsspec.register_implementation("oss", ossfs.OSSFileSystem, clobber=True)

# register customized HDFS implementation to make it seekable
import pyarrow
import pyarrow.fs


class ArrowFile(fsspec.implementations.arrow.ArrowFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def size(self):
        return self.stream.size()


class HDFSFileSystem(fsspec.implementations.arrow.HadoopFileSystem):
    @fsspec.implementations.arrow.wrap_exceptions
    def _open(self, path, mode="rb", block_size=None, **kwargs):
        if mode == "rb":
            method = self.fs.open_input_file
        elif mode == "wb":
            method = self.fs.open_output_stream
        elif mode == "ab":
            method = self.fs.open_append_stream
        else:
            raise ValueError(f"unsupported mode for Arrow filesystem: {mode!r}")

        _kwargs = {}
        if fsspec.implementations.arrow.PYARROW_VERSION[0] >= 4:
            # disable compression auto-detection
            _kwargs["compression"] = None
        if mode == 'rb':
            stream = method(path)
        else:
            stream = method(path, **_kwargs)

        return ArrowFile(self, stream, path, mode, block_size, **kwargs)


fsspec.register_implementation("hdfs", HDFSFileSystem, clobber=True)
