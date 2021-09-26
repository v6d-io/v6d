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

import contextlib
import logging
import pkg_resources
import subprocess
import sys
import textwrap
import time

from .utils import start_etcd, ssh_base_cmd

logger = logging.getLogger('vineyard')


@contextlib.contextmanager
def start_vineyardd(hosts=None,
                    etcd_endpoints=None,
                    vineyardd_path=None,
                    size='256M',
                    socket='/var/run/vineyard.sock',
                    rpc_socket_port=9600,
                    debug=False):
    ''' Launch a local vineyard cluster in a distributed fashion.

    Parameters:
        hosts: list of str
            A list of machines to launch vineyard server.
        etcd_endpoint: str
            Launching vineyard using specified etcd endpoints. If not specified, vineyard
            will launch its own etcd instance.
        vineyardd_path: str
            Location of vineyard server program. If not specified, vineyard will use its
            own bundled vineyardd binary.
        size: int
            The memory size limit for vineyard's shared memory. The memory size can be a plain
            integer or as a fixed-point number using one of these suffixes:

            .. code::

                E, P, T, G, M, K.

            You can also use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.

            For example, the following represent roughly the same value:

            .. code::

                128974848, 129k, 129M, 123Mi, 1G, 10Gi, ...
        socket: str
            The UNIX domain socket socket path that vineyard server will listen on.
        rpc_socket_port: int
            The port that vineyard will use to privode RPC service.
        debug: bool
            Whether print debug logs.
    '''
    if vineyardd_path is None:
        vineyardd_path = pkg_resources.resource_filename('vineyard', 'vineyardd')

    if hosts is None:
        hosts = ['localhost']

    if etcd_endpoints is None:
        etcd_ctx = start_etcd(host=hosts[0])
        etcd_proc, etcd_endpoints = etcd_ctx.__enter__()  # pylint: disable=no-member
    else:
        etcd_ctx = None

    env = dict()
    if debug:
        env['GLOG_v'] = 11

    # yapf: disable
    command = [
        vineyardd_path,
        '--deployment', 'distributed',
        '--size', str(size),
        '--socket', socket,
        '--rpc_socket_port', str(rpc_socket_port),
        '--etcd_endpoint', etcd_endpoints
    ]
    # yapf: enable

    procs = []
    try:
        for host in hosts:
            proc = subprocess.Popen(ssh_base_cmd(host) + command,
                                    env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=sys.__stderr__,
                                    universal_newlines=True,
                                    encoding='utf-8')
            procs.append(proc)
        time.sleep(1)
        for proc in procs:
            rc = proc.poll()
            if rc is not None:
                err = textwrap.indent(proc.stdout.read(), ' ' * 4)
                raise RuntimeError('vineyardd exited unexpectedly with code %d: error is:\n%s' % (rc, err))
        yield procs, socket, etcd_endpoints
    finally:
        logger.info('Distributed vineyardd being killed')
        for proc in procs:
            if proc.poll() is None:
                logger.info('killing the vineyardd')
                proc.kill()
        if etcd_ctx is not None:
            etcd_ctx.__exit__(None, None, None)  # pylint: disable=no-member


__all__ = ['start_vineyardd']
