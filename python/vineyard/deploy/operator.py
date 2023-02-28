#! /usr/bin/env python
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

import logging
import os
import subprocess

import pkg_resources

backup_image = 'ghcr.io/v6d-io/v6d/backup-job'
recover_image = 'ghcr.io/v6d-io/v6d/recover-job'
dask_repartition_image = 'ghcr.io/v6d-io/v6d/dask-repartition'
local_assembly_image = 'ghcr.io/v6d-io/v6d/local-assembly'
distributed_assembly_image = 'ghcr.io/v6d-io/v6d/distributed-assembly'


def create_vineyardd_cluster_with_operator(
    vineyardctl_path=None,
    namespace=None,
    kubeconfig=None,
    vineyard_replicas=3,
    vineyard_create_serviceAccount=False,
    vineyard_serviceAccount_name=None,
    vineyard_etcd_replicas=3,
    vineyard_name='vineyardd-sample',
    vineyard_container_image='vineyardcloudnative/vineyardd:latest',
    vineyard_container_image_pull_policy='IfNotPresent',
    vineyard_container_envs=None,
    vineyard_service_type='ClusterIP',
    vineyard_service_port=9600,
    vineyard_service_selector='rpc.vineyardd.v6d.io/rpc=vineyard-rpc',
    vineyard_metric_enable=False,
    vineyard_metric_image='vineyardcloudnative/vineyard-grok-exporter:latest',
    vineyard_metric_image_pull_policy='IfNotPresent',
    vineyardd_size='512Mi',
    vineyardd_socket='/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}',
    vineyardd_syncCRDs=True,
    vineyardd_streamThreshold=80,
    vineyardd_etcdEndpoint='http://etcd-for-vineyard:2379',
    vineyardd_etcdPrefix='/vineyard',
    vineyardd_spill_name=None,
    vineyardd_spill_path=None,
    vineyardd_spill_lowerRate=0.3,
    vineyardd_spill_upperRate=0.8,
    vineyardd_spill_pv=None,
    vineyardd_spill_pvc=None,
    vineyard_volume_pvcName=None,
    vineyard_volume_mountPath=None,
    vineyard_operator_plugin_backupImage=backup_image,
    vineyard_operator_plugin_recoverImage=recover_image,
    vineyard_operator_plugin_daskRepartitionImage=dask_repartition_image,
    vineyard_operator_plugin_localAssemblyImage=local_assembly_image,
    vineyard_operator_plugin_distributedAssemblyImage=distributed_assembly_image,
):
    """Launch a vineyard cluster with vineyard operator on kubernetes.

    Parameters:
        vineyardctl_path: str
            the path to the vineyardctl binary.
        namespace: str
            the namespace to launch vineyard cluster.
        kubeconfig: str
            the path to the kubeconfig file.
        vineyard_replicas: int
            the number of vineyardd instances.
        vineyard_create_serviceAccount: bool
            whether to create a service account for vineyard cluster.
        vineyard_serviceAccount_name: str
            the name of the service account for vineyard cluster.
        vineyard_etcd_replicas: int
            the number of etcd instances.
        vineyard_name: str
            the name of the vineyard cluster.
        vineyard_container_image: str
            the container image of vineyardd.
        vineyard_container_image_pull_policy: str
            the image pull policy of vineyardd.
        vineyard_container_envs: dict
            the environment variables of vineyardd.
        vineyard_service_type: str
            the service type of vineyardd.
        vineyard_service_port: int
            the service port of vineyardd.
        vineyard_service_selector: str
            the service selector of vineyardd.
        vineyard_metric_enable: bool
            whether to enable vineyard metrics.
        vineyard_metric_image: str
            the container image of vineyard metrics.
        vineyard_metric_image_pull_policy: str
            the image pull policy of vineyard metrics.
        vineyardd_size: str
            the memory size limit for vineyard's shared memory.
        vineyardd_socket: str
            the UNIX domain socket socket path that vineyard
            server will listen on.
        vineyardd_syncCRDS: bool
            whether to sync the CRDs for intermediate objects.
        vineyardd_streamThreshold: int
            memory threshold of streams (percentage of total memory)
        vineyardd_etcdEndpoint: str
            the etcd endpoint of vineyardd.
        vineyardd_etcdPrefix: str
            the etcd prefix of vineyardd.
        vineyardd_spill_name: str
            the name of the spill. If you want to enable the spill
            mechanism, you need to specify a name of the spill.
        vineyardd_spill_path: str
            the path of the spill.
        vineyardd_spill_lowerRate: float
            the lower rate of the spilling memory.
        vineyardd_spill_upperRate: float
            the upper rate of the spilling memory.
        vineyardd_spill_pv: str
            the name of the persistent volume for spilling. If you
            want to use the persistent volume for spilling, you need
            to specify a persistent volume.

            For example, a persistent volume is a json type of
            corev1.PersistentVolumeSpec as follows:

            .. code::

            '{
                "capacity":
                {
                    "storage":"1Gi"
                },
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": "manual",
                "hostPath": {"path": "/var/vineyard/dump"}
            }'

        vineyardd_spill_pvc: str
            the name of the persistent volume claim for spilling. If you
            want to use the persistent volume claim for spilling, you need
            to specify a persistent volume claim.

            For example, a persistent volume claim is a json type of
            corev1.PersistentVolumeClaimSpec as follows:

            .. code::

            '{
                "accessModes": ["ReadWriteOnce"],
                "resources":
                {
                    "requests":
                    {
                        "storage": "1Gi"
                    }
                }
            }'

        vineyard_volume_pvcName: str
            the name of the persistent volume claim for vineyardd
            such as socket.
        vineyard_volume_mountPath: str
            the mount path of the persistent volume claim for vineyardd
            such as socket.
        vineyard_operator_plugin_backupImage: str
            the image of the backup plugin.
        vineyard_operator_plugin_recoverImage: str
            the image of the recover plugin.
        vineyard_operator_plugin_daskRepartitionImage: str
            the image of the dask repartition plugin.
        vineyard_operator_plugin_localAssemblyImage: str
            the image of the local assembly plugin.
        vineyard_operator_plugin_distributedAssemblyImage: str
            the image of the distributed assembly plugin.
    """
    if vineyardctl_path is None:
        vineyardctl_path = pkg_resources.resource_filename('vineyard', 'vineyardctl')

    if not vineyardctl_path:
        raise RuntimeError('Unable to find the "vineyardctl" executable')

    command = [
        vineyardctl_path,
        'deploy',
        'vineyardd',
        '--name',
        vineyard_name,
        '--vineyard.replicas',
        str(vineyard_replicas),
        '--vineyard.etcd.replicas',
        str(vineyard_etcd_replicas),
        '--vineyard.image',
        vineyard_container_image,
        '--vineyard.imagePullPolicy',
        vineyard_container_image_pull_policy,
        '--vineyardd.service.type',
        vineyard_service_type,
        '--vineyardd.service.port',
        str(vineyard_service_port),
        '--vineyardd.service.selector',
        vineyard_service_selector,
        '--metric.image',
        vineyard_metric_image,
        '--metric.imagePullPolicy',
        vineyard_metric_image_pull_policy,
        '--vineyard.size',
        vineyardd_size,
        '--vineyard.socket',
        vineyardd_socket,
        '--vineyard.streamThreshold',
        str(vineyardd_streamThreshold),
        '--vineyard.etcdEndpoint',
        vineyardd_etcdEndpoint,
        '--vineyard.etcdPrefix',
        vineyardd_etcdPrefix,
        '--vineyard.spill.spillLowerRate',
        str(vineyardd_spill_lowerRate),
        '--vineyard.spill.spillUpperRate',
        str(vineyardd_spill_upperRate),
        '--plugin.backupImage',
        vineyard_operator_plugin_backupImage,
        '--plugin.recoverImage',
        vineyard_operator_plugin_recoverImage,
        '--plugin.daskRepartitionImage',
        vineyard_operator_plugin_daskRepartitionImage,
        '--plugin.localAssemblyImage',
        vineyard_operator_plugin_localAssemblyImage,
        '--plugin.distributedAssemblyImage',
        vineyard_operator_plugin_distributedAssemblyImage,
    ]

    if namespace:
        command.extend(['--namespace', namespace])
    if kubeconfig:
        command.extend(['--kubeconfig', kubeconfig])
    if vineyard_create_serviceAccount:
        command.extend(['--vineyard.create.serviceAccount'])
    if vineyard_serviceAccount_name:
        command.extend(['--vineyard.serviceAccount.name', vineyard_serviceAccount_name])
    if vineyard_container_envs:
        command.extend(['--envs', vineyard_container_envs])
    if vineyard_metric_enable:
        command.extend(['--metric.enable'])
    if vineyardd_syncCRDs:
        command.extend(['--vineyard.syncCRDs'])
    if vineyardd_spill_name:
        command.extend(['--vineyard.spill.config', vineyardd_spill_name])
    if vineyardd_spill_path:
        command.extend(['--vineyard.spill.path', vineyardd_spill_path])
    if vineyardd_spill_pv:
        command.extend(['--vineyard.spill.pv', vineyardd_spill_pv])
    if vineyardd_spill_pvc:
        command.extend(['--vineyard.spill.pvc', vineyardd_spill_pvc])
    if vineyard_volume_pvcName:
        command.extend(['--vineyard.volume.pvcname', vineyard_volume_pvcName])
    if vineyard_volume_mountPath:
        command.extend(['--vineyard.volume.mountPath', vineyard_volume_mountPath])

    print('command is:', command, flush=True)
    try:
        subprocess.Popen(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'Failed to create vineyard cluster with operator: %s' % e.stderr, flush=True
        )


def delete_vineyardd_cluster_with_operator(
    vineyardctl_path=None,
    namespace=None,
    kubeconfig=None,
    vineyard_name='vineyardd-sample',
):
    if vineyardctl_path is None:
        vineyardctl_path = pkg_resources.resource_filename('vineyard', 'vineyardctl')

    if not vineyardctl_path:
        raise RuntimeError('Unable to find the "vineyardctl" executable')

    command = [
        vineyardctl_path,
        'delete',
        'vineyardd',
        '--name',
        vineyard_name,
    ]
    if namespace:
        command.extend(['--namespace', namespace])
    if kubeconfig:
        command.extend(['--kubeconfig', kubeconfig])

    try:
        subprocess.Popen(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'Failed to delete vineyard cluster with operator: %s' % e.stderr
        )


def create_vineyardd_cluster_without_operator(
    vineyardctl_path=None,
    namespace=None,
    kubeconfig=None,
    label=None,
    vineyard_replicas=3,
    vineyard_create_serviceAccount=False,
    vineyard_serviceAccount_name=None,
    vineyard_etcd_replicas=3,
    vineyard_name='vineyardd-sample',
    vineyard_container_image='vineyardcloudnative/vineyardd:latest',
    vineyard_container_image_pull_policy='IfNotPresent',
    vineyard_container_envs=None,
    vineyard_service_type='ClusterIP',
    vineyard_service_port=9600,
    vineyard_service_selector='rpc.vineyardd.v6d.io/rpc=vineyard-rpc',
    vineyard_metric_enable=False,
    vineyard_metric_image='vineyardcloudnative/vineyard-grok-exporter:latest',
    vineyard_metric_image_pull_policy='IfNotPresent',
    vineyardd_size='512Mi',
    vineyardd_socket='/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}',
    vineyardd_syncCRDs=True,
    vineyardd_streamThreshold=80,
    vineyardd_etcdEndpoint='http://etcd-for-vineyard:2379',
    vineyardd_etcdPrefix='/vineyard',
    vineyardd_spill_name=None,
    vineyardd_spill_path=None,
    vineyardd_spill_lowerRate=0.3,
    vineyardd_spill_upperRate=0.8,
    vineyardd_spill_pv=None,
    vineyardd_spill_pvc=None,
    vineyard_volume_pvcName=None,
    vineyard_volume_mountPath=None,
):
    """Launch a vineyard cluster with vineyard operator on kubernetes.

    Parameters:
        vineyardctl_path: str
            the path to the vineyardctl binary.
        namespace: str
            the namespace to launch vineyard cluster.
        kubeconfig: str
            the path to the kubeconfig file.
        label: str
            the label to select the vineyard cluster.
            For example, a label is "app=vineyard-cluster",
            multiple labels are "app=vineyard-cluster,demo=test".
        vineyard_replicas: int
            the number of vineyardd instances.
        vineyard_create_serviceAccount: bool
            whether to create a service account for vineyard cluster.
        vineyard_serviceAccount_name: str
            the name of the service account for vineyard cluster.
        vineyard_etcd_replicas: int
            the number of etcd instances.
        vineyard_name: str
            the name of the vineyard cluster.
        vineyard_container_image: str
            the container image of vineyardd.
        vineyard_container_image_pull_policy: str
            the image pull policy of vineyardd.
        vineyard_container_envs: dict
            the environment variables of vineyardd.
        vineyard_service_type: str
            the service type of vineyardd.
        vineyard_service_port: int
            the service port of vineyardd.
        vineyard_service_selector: str
            the service selector of vineyardd.
        vineyard_metric_enable: bool
            whether to enable vineyard metrics.
        vineyard_metric_image: str
            the container image of vineyard metrics.
        vineyard_metric_image_pull_policy: str
            the image pull policy of vineyard metrics.
        vineyardd_size: str
            the memory size limit for vineyard's shared memory.
        vineyardd_socket: str
            the UNIX domain socket socket path that vineyard
            server will listen on.
        vineyardd_syncCRDs: bool
            whether to sync the CRDs for intermediate objects.
        vineyardd_streamThreshold: int
            memory threshold of streams (percentage of total memory)
        vineyardd_etcdEndpoint: str
            the etcd endpoint of vineyardd.
        vineyardd_etcdPrefix: str
            the etcd prefix of vineyardd.
        vineyardd_spill_name: str
            the name of the spill. If you want to enable the spill
            mechanism, you need to specify a name of the spill.
        vineyardd_spill_path: str
            the path of the spill.
        vineyardd_spill_lowerRate: float
            the lower rate of the spilling memory.
        vineyardd_spill_upperRate: float
            the upper rate of the spilling memory.
        vineyardd_spill_pv: str
            the name of the persistent volume for spilling. If you
            want to use the persistent volume for spilling, you need
            to specify a persistent volume.

            For example, a persistent volume is a json type of
            corev1.PersistentVolumeSpec as follows:

            .. code::

            '{
                "capacity":
                {
                    "storage":"1Gi"
                },
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": "manual",
                "hostPath": {"path": "/var/vineyard/dump"}
            }'

        vineyardd_spill_pvc: str
            the name of the persistent volume claim for spilling. If you
            want to use the persistent volume claim for spilling, you need
            to specify a persistent volume claim.

            For example, a persistent volume claim is a json type of
            corev1.PersistentVolumeClaimSpec as follows:

            .. code::

            '{
                "accessModes": ["ReadWriteOnce"],
                "resources":
                {
                    "requests":
                    {
                        "storage": "1Gi"
                    }
                }
            }'

        vineyard_volume_pvcName: str
            the name of the persistent volume claim for vineyardd
            such as socket.
        vineyard_volume_mountPath: str
            the mount path of the persistent volume claim for vineyardd
            such as socket.
    """
    if vineyardctl_path is None:
        vineyardctl_path = pkg_resources.resource_filename('vineyard', 'vineyardctl')

    if not vineyardctl_path:
        raise RuntimeError('Unable to find the "vineyardctl" executable')

    command = [
        vineyardctl_path,
        'dryapply',
        'vineyardd',
        '--name',
        vineyard_name,
        '--vineyard.replicas',
        str(vineyard_replicas),
        '--vineyard.etcd.replicas',
        str(vineyard_etcd_replicas),
        '--vineyard.image',
        vineyard_container_image,
        '--vineyard.imagePullPolicy',
        vineyard_container_image_pull_policy,
        '--vineyardd.service.type',
        vineyard_service_type,
        '--vineyardd.service.port',
        str(vineyard_service_port),
        '--vineyardd.service.selector',
        vineyard_service_selector,
        '--metric.image',
        vineyard_metric_image,
        '--metric.imagePullPolicy',
        vineyard_metric_image_pull_policy,
        '--vineyard.size',
        vineyardd_size,
        '--vineyard.socket',
        vineyardd_socket,
        '--vineyard.streamThreshold',
        str(vineyardd_streamThreshold),
        '--vineyard.etcdEndpoint',
        vineyardd_etcdEndpoint,
        '--vineyard.etcdPrefix',
        vineyardd_etcdPrefix,
        '--vineyard.spill.spillLowerRate',
        str(vineyardd_spill_lowerRate),
        '--vineyard.spill.spillUpperRate',
        str(vineyardd_spill_upperRate),
    ]
    if namespace:
        command.extend(['--namespace', namespace])
    if kubeconfig:
        command.extend(['--kubeconfig', kubeconfig])
    if label:
        command.extend(['--label', label])
    if vineyard_create_serviceAccount:
        command.extend(['--vineyard.create.serviceAccount'])
    if vineyard_serviceAccount_name:
        command.extend(['--vineyard.serviceAccount.name', vineyard_serviceAccount_name])
    if vineyard_container_envs:
        command.extend(['--envs', vineyard_container_envs])
    if vineyard_metric_enable:
        command.extend(['--metric.enable'])
    if vineyardd_syncCRDs:
        command.extend(['--vineyard.syncCRDs'])
    if vineyardd_spill_name:
        command.extend(['--vineyard.spill.config', vineyardd_spill_name])
    if vineyardd_spill_path:
        command.extend(['--vineyard.spill.path', vineyardd_spill_path])
    if vineyardd_spill_pv:
        command.extend(['--vineyard.spill.pv', vineyardd_spill_pv])
    if vineyardd_spill_pvc:
        command.extend(['--vineyard.spill.pvc', vineyardd_spill_pvc])
    if vineyard_volume_pvcName:
        command.extend(['--vineyard.volume.pvcname', vineyard_volume_pvcName])
    if vineyard_volume_mountPath:
        command.extend(['--vineyard.volume.mountPath', vineyard_volume_mountPath])

    print('command is:', command, flush=True)
    try:
        subprocess.Popen(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'Failed to create vineyard cluster with operator: %s' % e.stderr, flush=True
        )


def delete_vineyardd_cluster_without_operator(
    vineyardctl_path=None,
    namespace=None,
    kubeconfig=None,
    vineyard_name='vineyardd-sample',
):
    if vineyardctl_path is None:
        vineyardctl_path = pkg_resources.resource_filename('vineyard', 'vineyardctl')

    if not vineyardctl_path:
        raise RuntimeError('Unable to find the "vineyardctl" executable')

    command = [
        vineyardctl_path,
        'drydelete',
        'vineyardd',
        '--name',
        vineyard_name,
    ]
    if namespace:
        command.extend(['--namespace', namespace])
    if kubeconfig:
        command.extend(['--kubeconfig', kubeconfig])

    try:
        subprocess.Popen(command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Failed to delete vineyard cluster without operator: %s' % e)


def schedule_workload_on_vineyardd_cluster(
    vineyardctl_path=None,
    kubeconfig=None,
    workload=None,
    vineyard_name=None,
    vineyard_namespace=None,
):
    if vineyardctl_path is None:
        vineyardctl_path = pkg_resources.resource_filename('vineyard', 'vineyardctl')

    if not vineyardctl_path:
        raise RuntimeError('Unable to find the "vineyardctl" executable')

    command = [
        vineyardctl_path,
        'dryschedule',
        'workload',
        '--resource',
        workload,
        '--vineyardd-name',
        vineyard_name,
        '--vineyardd-namespace',
        vineyard_namespace,
    ]
    if kubeconfig:
        command.extend(['--kubeconfig', kubeconfig])

    try:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = p.communicate()
        if err:
            raise RuntimeError(
                'Failed to get the scheduled workload on vineyard cluster: %s' % err
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError('Failed to schedule workload on vineyard cluster: %s' % e)

    return output


__all__ = [
    'create_vineyardd_cluster_with_operator',
    'delete_vineyardd_cluster_with_operator',
    'create_vineyardd_cluster_without_operator',
    'delete_vineyardd_cluster_without_operator',
    'schedule_workload_on_vineyardd_cluster',
]
