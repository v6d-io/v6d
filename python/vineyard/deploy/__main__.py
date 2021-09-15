#!/usr/bin/env python
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

import logging
import signal

from .local import start_vineyardd

logger = logging.getLogger('vineyard')


def deploy_vineyardd(etcd_endpoints=None, size='256Mi', socket=None, rpc=True, rpc_socket_port=9600):
    with start_vineyardd(etcd_endpoints=etcd_endpoints,
                         size=size,
                         socket=socket,
                         rpc=rpc,
                         rpc_socket_port=rpc_socket_port) as (_, socket, _):
        logger.info("Vineyard server is listening %s ...", socket)

        try:
            signal.pause()
        except KeyboardInterrupt:
            logger.info("Vineyard exitting ...")


if __name__ == '__main__':
    deploy_vineyardd()
