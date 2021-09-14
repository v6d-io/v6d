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

import atexit
import contextlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

from .etcd import start_etcd
from .utils import find_vineyardd_path, check_socket
from .._C import connect

logger = logging.getLogger('vineyard')


@contextlib.contextmanager
def start_vineyardd(etcd_endpoints=None,
                    etcd_prefix=None,
                    vineyardd_path=None,
                    size='256M',
                    socket=None,
                    rpc=True,
                    rpc_socket_port=9600,
                    debug=False):
    ''' Launch a local vineyard cluster.

    Parameters:
        etcd_endpoint: str
            Launching vineyard using specified etcd endpoints. If not specified, vineyard
            will launch its own etcd instance.
        etcd_prefix: str
            Specify a common prefix to establish a local vineyard cluster.
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
            Default is None.

            When the socket parameter is None, a random path under temporary directory will be
            generated and used.
        rpc_socket_port: int
            The port that vineyard will use to privode RPC service.
        debug: bool
            Whether print debug logs.

    Returns:
        (proc, socket):
            Yields a tuple with the subprocess as the first element and the UNIX-domain
            IPC socket as the second element.
    '''

    if not vineyardd_path:
        vineyardd_path = find_vineyardd_path()

    if not vineyardd_path:
        raise RuntimeError('Unable to find the "vineyardd" executable')

    if not socket:
        socketfp = tempfile.NamedTemporaryFile(delete=True, prefix='vineyard-', suffix='.sock')
        socket = socketfp.name
        socketfp.close()

    if etcd_endpoints is None:
        etcd_ctx = start_etcd()
        etcd_proc, etcd_endpoints = etcd_ctx.__enter__()  # pylint: disable=no-member
    else:
        etcd_ctx = None

    env = os.environ.copy()
    if debug:
        env['GLOG_v'] = 11

    # yapf: disable
    command = [
        vineyardd_path,
        '--deployment', 'local',
        '--size', str(size),
        '--socket', socket,
        '--rpc' if rpc else '--norpc',
        '--rpc_socket_port', str(rpc_socket_port),
        '--etcd_endpoint', etcd_endpoints
    ]
    # yapf: enable

    if etcd_prefix is not None:
        command.extend(('--etcd_prefix', etcd_prefix))

    try:
        proc = subprocess.Popen(command,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=sys.__stderr__,
                                universal_newlines=True,
                                encoding='utf-8')
        # wait for vineyardd ready: check the rpc port and ipc sockets
        rc = proc.poll()
        while rc is None:
            if check_socket(socket) and ((not rpc) or check_socket(('0.0.0.0', rpc_socket_port))):
                break
            time.sleep(1)
            rc = proc.poll()

        if rc is not None:
            err = textwrap.indent(proc.stdout.read(), ' ' * 4)
            raise RuntimeError('vineyardd exited unexpectedly with code %d, error is:\n%s' % (rc, err))

        logger.debug('vineyardd is ready.............')
        yield proc, socket, etcd_endpoints
    finally:
        logger.debug('Local vineyardd being killed')
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
        try:
            shutil.rmtree(socket)
        except:
            pass
        if etcd_ctx is not None:
            etcd_ctx.__exit__(None, None, None)  # pylint: disable=no-member


__default_instance_contexts = {}


def init(num_instances=1, **kw):
    '''
    Help new users to launch a local vineyardd instance and get a client as easy as possible

    In a clean enviroment, simply use:
    
    .. code::
    
        vineyard.init()
    
    It will launch a local vineyardd and return a connected client to the vineyardd.
    It will also setup the environment variable **VINEYARD_IPC_SOCKET**.

    For the case to establish a local vineyard cluster consists of multiple vineyardd instances.
    Use the **num_instances** parameter:

    .. code::
    
        client1, client2, client3 = vineyard.init(num_instances=3)

    In this case, three vineyardd instances will be launched.

    The init method can only be called once in a process, to get the established sockets or clients later
    in the process, use **get_default_socket** or **get_default_client** respectively.
    '''
    assert __default_instance_contexts == {}

    if 'VINEYARD_IPC_SOCKET' in os.environ:
        raise ValueError(
            'VINEYARD_IPC_SOCKET has already been set: %s, which means there might be a vineyard daemon already running locally',
            os.environ['VINEYARD_IPC_SOCKET'])

    etcd_endpoints = None
    etcd_prefix = f'vineyard_init_at_{time.time()}'
    for idx in range(num_instances):
        ctx = start_vineyardd(etcd_endpoints=etcd_endpoints, etcd_prefix=etcd_prefix, rpc=False, **kw)
        _, ipc_socket, etcd_endpoints = ctx.__enter__()
        client = connect(ipc_socket)
        __default_instance_contexts[ipc_socket] = (ctx, client)
        if not idx:
            os.environ['VINEYARD_IPC_SOCKET'] = ipc_socket

    return get_default_client()


def get_default_client():
    if not __default_instance_contexts:
        raise ValueError("Vineyard has not been initialized, use vineyard.init()")
    clients = [__default_instance_contexts[k][1] for k in __default_instance_contexts]
    return clients if len(clients) > 1 else clients[0]


def get_default_socket():
    if not __default_instance_contexts:
        raise ValueError("Vineyard has not been initialized, use vineyard.init() to launch vineyard daemons")
    sockets = __default_instance_contexts.keys()
    return sockets if len(sockets) > 1 else sockets[0]


def shutdown():
    global __default_instance_contexts
    for ipc_socket in reversed(__default_instance_contexts):
        __default_instance_contexts[ipc_socket][0].__exit__(None, None, None)
    __default_instance_contexts = {}
    os.environ.pop('VINEYARD_IPC_SOCKET')


@atexit.register
def __shutdown_handler():
    try:
        shutdown()
    except Exception:
        pass


__all__ = ['start_vineyardd']
