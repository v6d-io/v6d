#! /usr/bin/env python
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

import jinja2
import kubernetes
import os
import pkg_resources
import subprocess
import textwrap
import time

from .utils import start_etcd_k8s


def start_vineyardd(etcd_endpoints=None,
                    vineyardd_path=None,
                    size='256Mi',
                    socket='/var/run/vineyard.sock',
                    rpc_socket_port=9600,
                    debug=False):
    ''' Launch a local vineyard cluster.

    Parameters:
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
    if etcd_endpoints is None:
        start_etcd_k8s()

    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("vineyard_k8s.yaml.tmpl")
    body = template.render()

    kubernetes.config.load_kube_config()

    k8s_apps_v1 = kubernetes.client.AppsV1Api()
    resp = k8s_apps_v1.create_namespaced_daemon_set(
            body=body, namespace="vineyard")
    print("DaemonSet created. status=%s" % resp.metadata.name)
