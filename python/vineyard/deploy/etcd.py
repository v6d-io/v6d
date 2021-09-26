#!/usr/bin/env python3
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
import os
from posixpath import join
import shutil
import subprocess
from sys import path
import tempfile
import textwrap
import time

try:
    import kubernetes
except ImportError:
    kubernetes = None

from .utils import find_port, ssh_base_cmd, check_socket

logger = logging.getLogger('vineyard')


@contextlib.contextmanager
def start_etcd(host=None, etcd_executable=None, data_dir=None):
    if etcd_executable is None:
        etcd_executable = shutil.which('etcd')

    if host is None:
        srv_host = '127.0.0.1'
        client_port = find_port()
        peer_port = find_port()
    else:
        srv_host = host
        client_port = 2379
        peer_port = 2380

    if data_dir is None:
        with tempfile.TemporaryDirectory(prefix='etcd-') as td:
            data_dir_base = td
            data_dir = os.path.join(td, 'default.etcd')
    else:
        data_dir_base = None
        data_dir = os.path.join(data_dir, 'default.etcd')

    # yapf: disable
    prog_args = [
        etcd_executable,
        '--data-dir', data_dir,
        '--max-txn-ops=102400',
        '--listen-peer-urls', 'http://0.0.0.0:%d' % peer_port,
        '--listen-client-urls', 'http://0.0.0.0:%d' % client_port,
        '--advertise-client-urls', 'http://%s:%d' % (srv_host, client_port),
        '--initial-cluster', 'default=http://%s:%d' % (srv_host, peer_port),
        '--initial-advertise-peer-urls', 'http://%s:%d' % (srv_host, peer_port)
    ]
    # yapf: enable

    proc = None
    try:
        if host is None:
            commands = []
        else:
            commands = ssh_base_cmd(host)
        proc = subprocess.Popen(commands + prog_args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True,
                                encoding='utf-8')

        rc = proc.poll()
        while rc is None:
            if check_socket((host or '0.0.0.0', client_port)):
                break
            time.sleep(1)
            rc = proc.poll()

        if rc is not None:
            err = textwrap.indent(proc.stdout.read(), ' ' * 4)
            raise RuntimeError('Failed to launch program etcd on %s, unexpected error:\n%s' %
                               (srv_host or 'local', err))
        yield proc, 'http://%s:%d' % (srv_host, client_port)
    finally:
        logging.info('Etcd being killed...')
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
        try:
            if data_dir_base:
                shutil.rmtree(data_dir_base)
            else:
                shutil.rmtree(data_dir)
        except:
            pass


def start_etcd_k8s(namespace):
    if kubernetes is None:
        raise RuntimeError('Please install the kubernetes python first')
    kubernetes.config.load_kube_config()
    k8s_client = kubernetes.client.ApiClient()
    return kubernetes.utils.create_from_yaml(k8s_client,
                                             os.path.join(os.path.dirname(__file__), 'etcd.yaml'),
                                             namespace=namespace)
