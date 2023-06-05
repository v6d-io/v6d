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

import concurrent
import concurrent.futures
import json
import multiprocessing
import os
import traceback

from vineyard._C import ObjectID


def report_status(status, content):
    print(
        json.dumps(
            {
                'type': status,
                'content': content,
            }
        ),
        flush=True,
    )


def report_error(content):
    report_status('error', content)


def report_success(content):
    if isinstance(content, ObjectID):
        content = repr(content)
    report_status('return', content)


def report_exception():
    report_status('error', traceback.format_exc())


def expand_full_path(path):
    return os.path.expanduser(os.path.expandvars(path))


def parse_readable_size(size):
    """Parse human-readable size. Note that any extra character that follows a
    valid sequence will be ignored.

    You can express memory as a plain integer or as a fixed-point number
    using one of these suffixes: E, P, T, G, M, K. You can also use the
    power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.

    For example, the following represent roughly the same value:

    .. code:: python

        128974848, 129k, 129M, 123Mi, 1G, 10Gi, ...
    """
    if isinstance(size, (int, float)):
        return int(size)

    ns, cs = '', ''
    for c in size:
        if c.isdigit():
            ns += c
        else:
            cs = c.upper()
            break
    ratios = {
        'K': 2**10,
        'M': 2**20,
        'G': 2**30,
        'T': 2**40,
        'P': 2**50,
        'E': 2**60,
    }
    return int(ns) * ratios.get(cs, 1)


class capture_exception:
    """Capture the possible exception and throw later when `.check()` is called."""

    def __init__(self):
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is not None:
            self.exception = exc_type(exc_val).with_traceback(traceback)
        return True

    def check(self):
        if self.exception is not None:
            raise self.exception

    def print(self):
        if self.exception is not None:
            traceback.print_exception(
                type(self.exception), self.exception, self.exception.__traceback__
            )


class BaseStreamExecutor:
    def execute(self):
        """Execute the action on stream chunks."""


class ThreadStreamExecutor:
    def __init__(self, executor_cls, parallelism: int = None, **kwargs):
        if parallelism is None:
            self._parallelism = multiprocessing.cpu_count()
        else:
            self._parallelism = parallelism
        self._executors = [executor_cls(**kwargs) for _ in range(self._parallelism)]

    def execute(self):
        def start_to_execute(executor: BaseStreamExecutor):
            return executor.execute()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._parallelism
        ) as executor:
            results = [
                executor.submit(start_to_execute, exec) for exec in self._executors
            ]
            return [future.result() for future in results]
