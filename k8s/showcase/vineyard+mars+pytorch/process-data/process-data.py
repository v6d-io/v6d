import os
import time

import vineyard
import vineyard.io
import mars.dataframe as md
from mars.dataframe.datastore.to_vineyard import to_vineyard
from mars.dataframe.datasource.from_vineyard import from_vineyard


env_dist = os.environ

required_job = env_dist['REQUIRED_JOB_NAME']
allglobalobjects = env_dist[required_job]
globalobjects = allglobalobjects.split(',')

socket = '/var/run/vineyard.sock'
vineyard_client = vineyard.connect(socket)

def launch_on_k8s():
    import requests
    import urllib3
    requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'ALL:@SECLEVEL=1'

    from kubernetes import config
    from mars.deploy.kubernetes import new_cluster
    from mars.deploy.kubernetes.config import EmptyDirVolumeConfig, HostPathVolumeConfig

    # use in cluster config here
    k8sconfig=config.load_incluster_config()

    print('launching a mars cluster on kubernetes ...',flush=True)
    volumns = []
    volumns.append(EmptyDirVolumeConfig("vineyard", "/tmp/vineyard"))
    # use the default vineyard socket path
    volumns.append(HostPathVolumeConfig("run", "/var/run", "/var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample"))

    envs = {'VINEYARD_IPC_SOCKET': '/var/run/vineyard.sock', 'WITH_VINEYARD': 'True', 'JOB_NAME': 'process-data'}

    return new_cluster(k8sconfig,
            namespace='vineyard-app',
            image='ghcr.io/v6d-io/v6d/mars-with-vineyard:v0.10.0',
            worker_cpu=2,
            worker_mem=3 * 1024**3,
            worker_num=3,
            extra_volumes=volumns,
            extra_env=envs,
            mount_shm=False)

cluster = launch_on_k8s()
session = cluster.session.as_default()

dataid = []
for o in globalobjects:
    meta = vineyard_client.get_meta(vineyard.ObjectID(o))
    if meta['typename'] == 'vineyard::GlobalDataFrame':
        dataid.append(o)

for d in dataid:
    global_df = from_vineyard(vineyard.ObjectID(d),vineyard_socket=socket).execute(session=session)
    if "txnfeatures-0" in str(global_df.columns_value):
        txns = global_df.reset_index(drop=True)
    elif "itemfeature-0" in str(global_df.columns_value):
        items = global_df.reset_index(drop=True).set_index('id',drop=True)
    else:
        users = global_df.reset_index(drop=True).set_index('id',drop=True)

dataset = txns
dataset = dataset.join(users, how='left', on='user')
dataset = dataset.join(items, how='left', on='item')

global_df = to_vineyard(dataset).execute(session=session)
print(global_df,flush=True)