# pylint: disable=django-not-configured
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

limits = int(env_dist['LIMIT'])
path = env_dist['BACKUP_PATH']
socket = '/var/run/vineyard.sock'
endpoint = env_dist['ENDPOINT']
service = (endpoint, 9600)
client = vineyard.connect(socket)

# get instance id
instance = client.instance_id

objs = client.list_objects(pattern='*',limit=limits)

# serialize all persistent objects
for i in enumerate(objs):
    try:
        meta = client.get_meta(objs[i].id, sync_remote=True)
        if not meta['transient'] and meta['instance_id']==int(instance):
            objname = str(objs[i].id).split("\"")[1]
            objpath = path + '/' + objname
            print(objpath)
            if not os.path.exists(objpath):
                os.makedirs(objpath)
            vineyard.io.serialize(
                objpath,
                objs[i].id,
                vineyard_ipc_socket=socket,
                vineyard_endpoint=service,
            )
    except vineyard.ObjectNotExistsException:
        print(objs[i].id,' trigger ObjectNotExistsException')
