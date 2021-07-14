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

from enum import Enum
import json
import logging
import sys
import threading
import uuid

logger = logging.getLogger('vineyard')


class LauncherStatus(Enum):
    INITIAL = 0
    RUNNING = 1
    FAILED = 2
    SUCCEED = 3


class Launcher(object):
    ''' :code:`Launcher` is an abstraction about a managed vineyard job. :code:`Launcher`
        provides a unified interface for users to invoke a vineyard driver (internal or external).

        To implements a user-defined launcher, the developer needs to implements:

            - __init__: configure the launcher by ad-hoc paramters
            - launch: launch the job
            - wait: wait and block until the job complete, if the job has already finished, return
              the result immediately
            - dispose: interrupt the process and destroy the job
            - status: query the execution status of the job
    '''
    def __init__(self):
        self._id = uuid.uuid4()
        self._result = []  # a result variable queue
        self._result_cv = threading.Condition()
        self._status = LauncherStatus.INITIAL
        self._diagnostics = []

    @property
    def id(self):
        return self._id

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def wait(self, timeout=None):
        ''' Will block until obtain an result, a :code:`Launcher` can be "wait" for
            multiple since the job might yields many results.

            The launcher can be safely "wait" at the same time in multiple threads,
            the order of which thread will be weak up is unspecified.

            Note that :code:`wait` doesn't means the finish or exit of the job, it indicates
            the job has yield some results, and the job itself may or may not continue.
        '''

        with self._result_cv:
            if not self._result:
                if self._status != LauncherStatus.RUNNING:
                    raise RuntimeError('Cannot wait the the launcher that with status %s' % self._status)
                if not self._result_cv.wait(timeout=timeout):
                    return None
            if not self._result:  # if still no available result object, raise to users.
                raise RuntimeError('The job failed unexpectedly')
            return self._result.pop(0)

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def status(self):
        ''' Query the status of the launched job.
        '''
        return self._status

    def join(self):
        ''' Wait util the launched job exit.
        '''
        raise NotImplementedError

    def dispose(self, desired=True):
        ''' Dispose the launched job and release resources.

            Parameters
            ----------
            desired: bool
                Whether the dispose is an desired action.
        '''
        raise NotImplementedError

    def parse(self, line):
        ''' Parse job message. The messages should statisfy the following spec:

            ..code-block:

                {
                    "type": "return/error/exit",
                    "content": ......
                }

            or, in a more compact format:

                {
                    "return/error/exit": ......
                }
        '''
        try:
            line = line.strip()
            if line:
                logger.debug('receive output: %s', line)
            else:
                return None

            r = json.loads(line)
            if not isinstance(r, dict):
                return None

            if 'return' in r:
                return self.on_return(r['return'])
            if 'error' in r:
                return self.on_error(r['error'])
            if 'exit' in r:
                return self.on_exit(r['exit'])

            if 'type' in r:
                if r['type'] == 'return':
                    return self.on_return(r.get('content'))
                if r['type'] == 'error':
                    return self.on_error(r.get('content'))
                if r['type'] == 'exit':
                    return self.on_exit(r.get('content'))
            return None
        except json.JSONDecodeError:
            # if not a valid launcher message, silently ignore it.
            return None

    def on_return(self, return_content):
        ''' The on-return handle, can be overrided to process events with type "return".
        '''
        with self._result_cv:
            self._result.append(return_content)
            self._result_cv.notify()

    def on_error(self, error_content):
        ''' The on-error handle, can be overrided to process events with type "error".
        '''
        self._diagnostics.append(error_content)

    def on_exit(self, exit_content):
        ''' The on-exit handle, can be overrided to process events with type "exit".
        '''
        if exit_content and exit_content.strip():
            print(exit_content, file=sys.stderr)
        self.dispose(True)
