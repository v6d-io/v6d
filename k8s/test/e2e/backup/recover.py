#! /usr/bin/env python3
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

import os
import vineyard

env_dist = os.environ

path = env_dist['RECOVER_PATH']
socket = '/var/run/vineyard.sock'
endpoint = env_dist['ENDPOINT']
service = (endpoint, 9600)
client = vineyard.connect(socket)

objslist = []
if os.path.exists(path):
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path, file)
        if os.path.isdir(m):
            objslist.append(m)

for objs in objslist:
    obj = vineyard.io.deserialize(
        objs,
        vineyard_ipc_socket=socket,
        vineyard_endpoint=service,
    )
    oldobj = os.path.split(objs)
    newobj = str(obj).split("\"")[1]
    print(f'{oldobj[1]}->{newobj}',flush=True)
