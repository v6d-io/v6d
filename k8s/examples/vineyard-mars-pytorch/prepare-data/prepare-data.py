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
import vineyard.io

from kubernetes import client, config
from kubernetes.client.rest import ApiException

env_dist = os.environ
namespace = env_dist['POD_NAMESPACE']
podname = env_dist['POD_NAME']
selector = env_dist['SELECTOR']
allinstances = int(env_dist['ALLINSTANCES'])
endpoint = env_dist['ENDPOINT']
vineyard_endpoint = (endpoint, 9600)

config.load_incluster_config()
k8s_api_obj = client.CoreV1Api()
podlists = []
hosts = []
while True:
    api_response = k8s_api_obj.list_namespaced_pod(namespace, label_selector="app=" + selector)
    # wait all pods to be deployed on nodes
    alldeployedpods = 0
    for i in api_response.items:
        if i.spec.node_name is not None and i.status.phase == 'Running':
            alldeployedpods += 1
    if alldeployedpods == allinstances:
        for i in api_response.items:
            name = namespace + ":" + i.metadata.name
            hosts.append(name)
            podlists.append(i.metadata.name)
        break
    time.sleep(5)

# get instance id
socket = '/var/run/vineyard.sock'
vineyard_client = vineyard.connect(socket)
instance = vineyard_client.instance_id
print('instance id:',instance)

files = ["file:///datasets/user.csv", "file:///datasets/item.csv", "file:///datasets/txn.csv"]
print('opened ', files[instance%allinstances], flush=True)
values = vineyard.io.open(
        files[instance%allinstances],
        vineyard_ipc_socket='/var/run/vineyard.sock',
        vineyard_endpoint=vineyard_endpoint,
        type="global",
        hosts=hosts,
        deployment='kubernetes',
        read_options={"header_row": True, "delimiter": ","},
        accumulate=True,
        index_col=0
)

# wait other prepare data process ready
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
                print('Found exception in reading the logs', e)
    if succeedInstance == allinstances-1:
        succeed = False
        break
    time.sleep(10)
