import os
import vineyard.io
import time

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
        if i.spec.node_name is not None:
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

if instance%allinstances == 0:
    print('open user.csv',flush=True)
    users = vineyard.io.open(
        'file:///datasets/user.csv', 
        vineyard_ipc_socket='/var/run/vineyard.sock',
        vineyard_endpoint=vineyard_endpoint,
        type="global",
        hosts=hosts,
        deployment='kubernetes',
        read_options={"header_row": True, "delimiter": ","},
        accumulate=True,
        index_col=0
    )
if instance%allinstances == 1:
    print('open item.csv',flush=True)
    items = vineyard.io.open(
        'file:///datasets/item.csv',
        vineyard_ipc_socket='/var/run/vineyard.sock',
        vineyard_endpoint=vineyard_endpoint,
        type="global",
        hosts=hosts,
        deployment='kubernetes',
        read_options={"header_row": True, "delimiter": ","},
        accumulate=True,
        index_col=0
    )
if instance%allinstances == 2:
    print('open txn.csv',flush=True)
    txns = vineyard.io.open(
        'file:///datasets/txn.csv',
        vineyard_ipc_socket='/var/run/vineyard.sock',
        vineyard_endpoint=vineyard_endpoint,
        type="global",
        hosts=hosts,
        deployment='kubernetes',
        read_options={"header_row": True, "delimiter": ","},
        accumulate=True
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
