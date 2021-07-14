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

import kubernetes
import logging
import os
import pkg_resources
import subprocess
import tempfile
import textwrap
import time
import yaml

from .etcd import start_etcd_k8s

logger = logging.getLogger('vineyard')


def start_vineyardd(namespace='vineyard', size='256Mi', socket='/var/run/vineyard.sock', rpc_socket_port=9600):
    ''' Launch a vineyard cluster on kubernetes.

    Parameters:
        namespace: str
            namespace in kubernetes
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
    '''
    start_etcd_k8s(namespace)

    with open(os.path.join(os.path.dirname(__file__), "vineyard.yaml.tpl"), 'r', encoding='utf-8') as fp:
        definitions = fp.read().format(Namespace=namespace, Size=size, Socket=socket, Port=rpc_socket_port)

    rendered = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
    try:
        rendered.write(definitions)
        rendered.flush()
        rendered.close()

        with open(rendered.name, 'r', encoding='utf-8') as fp:
            logger.debug(fp.read())

        kubernetes.config.load_kube_config()
        k8s_client = kubernetes.client.ApiClient()
        return kubernetes.utils.create_from_yaml(k8s_client, rendered.name, namespace=namespace)
    finally:
        os.unlink(rendered.name)


__all__ = ['start_vineyardd']
