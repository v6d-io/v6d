#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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
import time
import vineyard
from kubernetes import client, config
from kubernetes.client.rest import ApiException

env_dist = os.environ

objectids = None
if "ObjectIDs" in env_dist:
    objectids = env_dist["ObjectIDs"]
path = env_dist['BACKUP_PATH']
socket = '/var/run/vineyard.sock'
endpoint = env_dist['ENDPOINT']
service = (endpoint, 9600)
vineyard_client = vineyard.connect(socket)

allinstances = int(env_dist['ALLINSTANCES'])
# get pod name
selector = env_dist['SELECTOR']
podname = env_dist['POD_NAME']
namespace = env_dist['POD_NAMESPACE']
config.load_incluster_config()

k8s_api_obj = client.CoreV1Api()
hosts = []
podlists = []
while True:
    api_response = k8s_api_obj.list_namespaced_pod(namespace, label_selector="app.kubernetes.io/name=" + selector)
    # wait all pods to be deployed on nodes
    alldeployedpods = 0
    for i in api_response.items:
        if i.spec.node_name is not None and i.status.phase == 'Running':
            alldeployedpods += 1
    if alldeployedpods == allinstances:
        api_response.items.sort(key=lambda a: a.spec.node_name)
        for i in api_response.items:
            podlists.append(i.metadata.name)
            hosts.append(namespace + ':' + i.metadata.name)
        break
    time.sleep(5)

# get instance id
instance = vineyard_client.instance_id

objIDs = []
if objectids is not None:
    objs = objectids.split(',')
    for _,o in enumerate(objs):
        objIDs.append(vineyard.ObjectID(o))
else:
    objs = vineyard_client.list_objects(pattern='*')
    for _,o in enumerate(objs):
        objIDs.append(o.id)


# serialize all persistent objects
for _,o in enumerate(objIDs):
    try:
        meta = vineyard_client.get_meta(o, sync_remote=True)
        if not meta['transient'] and meta['typename'] not in ['vineyard::ParallelStream','vineyard::StreamCollection']:
            objname = str(o).split("\"")[1]
            objpath = path + '/' + objname
            # for local objects
            if not meta['global'] and meta['instance_id']==instance:
                objpath += '-local-' + str(instance)
                if not os.path.exists(objpath):
                    os.makedirs(objpath)
                vineyard.io.serialize(
                    objpath,
                    o,
                    vineyard_ipc_socket=socket,
                    vineyard_endpoint=service,
                )
            # for global objects
            if meta['global'] and meta['instance_id']%allinstances==instance:
                objpath += '-global-' + str(instance)
                if not os.path.exists(objpath):
                    os.makedirs(objpath)
                vineyard.io.serialize(
                    objpath,
                    o,
                    type="global",
                    vineyard_ipc_socket=socket,
                    vineyard_endpoint=service,
                    deployment='kubernetes',
                    hosts=hosts,
                )
    except vineyard.ObjectNotExistsException:
        print(o,' trigger ObjectNotExistsException')

# wait other backup process ready
print('Succeed',flush=True)

succeed = True
while succeed:
    succeedInstance = 0
    for p in podlists:
        if p != podname:
            try:
                api_response = k8s_api_obj.read_namespaced_pod_log(name=p,namespace=namespace)
                if "Succeed" in api_response:
                    succeedInstance += 1
            except ApiException as e:
                print('Found exception in reading the logs')
    if succeedInstance == allinstances-1:
        succeed = False
        break
    time.sleep(10)
