#!/usr/bin/env python3
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

import contextlib
import logging
import os
import shutil
import socket
import subprocess
import textwrap
import time

try:
    import kubernetes
except ImportError:
    kubernetes = None

logger = logging.getLogger('vineyard')


def ssh_base_cmd(host):
    return [
        'ssh', host, '--', 'shopt', '-s', 'huponexit', '2>/dev/null', '||', 'setopt', 'HUP', '2>/dev/null', '||',
        'true;'
    ]


def find_port_probe(start=2048, end=20480):
    ''' Find an available port in range [start, end)
    '''
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                yield port


ipc_port_finder = find_port_probe()


def find_port():
    return next(ipc_port_finder)


__vineyardd_path = None


def find_vineyardd_path():
    global __vineyardd_path

    if __vineyardd_path is not None:
        return __vineyardd_path

    vineyardd_path = shutil.which('vineyardd')
    if vineyardd_path is None:
        if 'VINEYARD_HOME' in os.environ:
            vineyardd_path = os.path.expandvars('$VINEYARD_HOME/vineyardd')
    if vineyardd_path is not None:
        if not (os.path.isfile(vineyardd_path) and os.access(vineyardd_path, os.R_OK)):
            vineyardd_path = None
    if vineyardd_path is None:
        return None

    __vineyardd_path = vineyardd_path
    return vineyardd_path


@contextlib.contextmanager
def start_etcd(host=None, etcd_executable=None):
    if etcd_executable is None:
        etcd_executable = '/usr/local/bin/etcd'
    if host is None:
        srv_host = '127.0.0.1'
        client_port = find_port()
        peer_port = find_port()
    else:
        srv_host = host
        client_port = 2379
        peer_port = 2380

    # yapf: disable
    prog_args = [
        etcd_executable,
        '--max-txn-ops=102400',
        '--listen-peer-urls', 'http://0.0.0.0:%d' % peer_port,
        '--listen-client-urls', 'http://0.0.0.0:%d' % client_port,
        '--advertise-client-urls', 'http://%s:%d' % (srv_host, client_port),
        '--initial-cluster', 'default=http://%s:%d' % (srv_host, peer_port),
        '--initial-advertise-peer-urls', 'http://%s:%d' % (srv_host, peer_port)
    ]
    # yapf: enable

    try:
        proc = None
        if host is None:
            commands = []
        else:
            commands = ssh_base_cmd(host)
        proc = subprocess.Popen(commands + prog_args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True,
                                encoding='utf-8')
        time.sleep(1)
        if proc.poll() is not None:
            err = textwrap.indent(proc.stdout.read(), ' ' * 4)
            raise RuntimeError('Failed to launch program etcd on %s, error:\n%s' % (srv_host, err))
        yield proc, 'http://%s:%d' % (srv_host, client_port)
    finally:
        logging.info('Etcd being killed...')
        if proc is not None and proc.poll() is None:
            proc.terminate()


def start_etcd_k8s(namespace):
    if kubernetes is None:
        raise RuntimeError('Please install the kubernetes python first')
    kubernetes.config.load_kube_config()
    k8s_client = kubernetes.client.ApiClient()
    resp = kubernetes.utils.create_from_yaml(k8s_client,
                                             os.path.join(os.path.dirname(__file__), 'etcd.yaml'),
                                             namespace=namespace)
