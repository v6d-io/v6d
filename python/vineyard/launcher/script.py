#! /usr/bin/env python
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

import logging
import os
import subprocess
import threading

from .launcher import Launcher
from .launcher import LauncherStatus

logger = logging.getLogger('vineyard')


class ScriptLauncher(Launcher):  # pylint: disable=too-many-instance-attributes
    """Launch the job by executing a script.

    The output of script must be printed to stdout, rather than stderr.
    """

    def __init__(self, script):
        super().__init__()
        self._script = script
        self._proc = None
        self._listen_out_thrd = None
        self._listen_err_thrd = None
        self._cmd = None

        self._err_message = None
        self._exit_code = None
        self._stopped = False

    @property
    def command(self):
        return self._cmd

    @property
    def error_message(self):
        return self._err_message

    @property
    def exit_code(self):
        return self._exit_code

    def run(self, *args, **kw):
        # FIXME run self._script on a set of host machines, the host is decided
        # by the arguments of the launcher in `__init__`, and those inputs object
        cmd = [self._script]
        for arg in args:
            if isinstance(arg, str):
                cmd.append(arg.encode('unicode-escape').decode('utf-8'))
            else:
                cmd.append(repr(arg))
        env = os.environ.copy()
        for key, value in kw.items():
            # if key is all in lower cases, treat it as arguments, otherwise as the
            # environment variables.
            if key.islower():
                cmd.append('--%s' % key)
                if isinstance(value, str):
                    cmd.append(value)
                else:
                    cmd.append(repr(value))
            else:
                env[key] = value
        logger.debug('command is: %s', ' '.join(cmd))
        self._cmd = cmd
        self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd,
            env=env,
            universal_newlines=True,
            encoding='utf-8',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._status = LauncherStatus.RUNNING

        self._listen_out_thrd = threading.Thread(
            target=self.read_output, args=(self._proc.stdout,)
        )
        self._listen_out_thrd.daemon = True
        self._listen_out_thrd.start()
        self._err_message = []
        self._listen_err_thrd = threading.Thread(
            target=self.read_err, args=(self._proc.stderr,)
        )
        self._listen_err_thrd.daemon = True
        self._listen_err_thrd.start()

    def wait(self, timeout=None):
        # a fast wait: to use existing response directly, since the io adaptor
        # may finish immediately.
        r = super().wait(timeout=0)
        if r is not None:
            return r
        elapsed, period = 0, 1
        while self._proc.poll() is None:
            if timeout is not None and elapsed > timeout:
                raise TimeoutError(
                    'Unable to wait for status of job [%s] after %r seconds'
                    % (self._cmd, timeout)
                )
            r = super().wait(timeout=period)
            elapsed += period
            if r is None:
                continue
            else:
                return r
        r = super().wait(timeout=period)
        if r is not None:
            return r
        remaining = self._proc.stdout.read()
        if remaining:
            for line in remaining.split('\n'):
                self.parse(line)
        r = super().wait(timeout=period)
        if r is not None:
            return r
        if isinstance(self._cmd, list):
            cmd = ' '.join(self._cmd)
        else:
            cmd = self._cmd
        raise RuntimeError(
            'Failed to launch job [%s], exited with %r: %s'
            % (cmd, self._proc.poll(), ''.join(self._err_message))
        )

    def read_output(self, stdout):
        while self._proc.poll() is None:
            line = stdout.readline()
            if line:
                self.parse(line)

        # consume all extra lines if the proc exits.
        for line in stdout.readlines():
            if line:
                self.parse(line)

    def read_err(self, stderr):
        while self._proc.poll() is None:
            line = stderr.readline()
            if line:
                self._err_message.append(line)
        self._err_message.extend(stderr.readlines())

    def join(self):
        if self._stopped:
            return
        self._stopped = True

        self._exit_code = self._proc.wait()
        if self._exit_code != 0:
            self._status = LauncherStatus.FAILED
        else:
            self._status = LauncherStatus.SUCCEED

        # makes the listen thread exits.
        self._listen_out_thrd.join()
        self._listen_err_thrd.join()

    def dispose(self, desired=True):
        if self._stopped:
            return None
        self._stopped = True

        if self._status == LauncherStatus.RUNNING:
            try:
                self._proc.terminate()
            except ProcessLookupError:
                pass
            if desired:
                self._status = LauncherStatus.FAILED
            else:
                self._status = LauncherStatus.SUCCEED
        return self._status
