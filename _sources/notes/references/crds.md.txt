# API Reference

## Packages
- [k8s.v6d.io/v1alpha1](#k8sv6diov1alpha1)


## k8s.v6d.io/v1alpha1

Package v1alpha1 contains API Schema definitions for the k8s v1alpha1 API group

### Resource Types
- [Backup](#backup)
- [BackupList](#backuplist)
- [CSIDriver](#csidriver)
- [CSIDriverList](#csidriverlist)
- [GlobalObject](#globalobject)
- [GlobalObjectList](#globalobjectlist)
- [LocalObject](#localobject)
- [LocalObjectList](#localobjectlist)
- [Operation](#operation)
- [OperationList](#operationlist)
- [Recover](#recover)
- [RecoverList](#recoverlist)
- [Sidecar](#sidecar)
- [SidecarList](#sidecarlist)
- [Vineyardd](#vineyardd)
- [VineyarddList](#vineyarddlist)



#### Backup



Backup describes a backup operation of vineyard objects, which uses the [Kubernetes PersistentVolume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) to store the backup data. Every backup operation will be binded with the name of Backup.

_Appears in:_
- [BackupList](#backuplist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `Backup`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[BackupSpec](#backupspec)_ |  |


#### BackupList



BackupList contains a list of Backup



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `BackupList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[Backup](#backup) array_ |  |


#### BackupSpec



BackupSpec defines the desired state of Backup

_Appears in:_
- [Backup](#backup)

| Field | Description |
| --- | --- |
| `vineyarddName` _string_ | the name of the vineyard cluster |
| `vineyarddNamespace` _string_ | the namespace of the vineyard cluster |
| `objecIDs` _string array_ | the specific objects to be backed up if not specified, all objects will be backed up |
| `backupPath` _string_ | the path of backup data |
| `persistentVolumeSpec` _[PersistentVolumeSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#persistentvolumespec-v1-core)_ | the PersistentVolumeSpec of the backup data |
| `persistentVolumeClaimSpec` _[PersistentVolumeClaimSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#persistentvolumeclaimspec-v1-core)_ | the PersistentVolumeClaimSpec of the backup data |




#### CSIDriver



CSIDriver is the Schema for the csidrivers API

_Appears in:_
- [CSIDriverList](#csidriverlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `CSIDriver`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[CSIDriverSpec](#csidriverspec)_ |  |


#### CSIDriverList



CSIDriverList contains a list of CSIDriver



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `CSIDriverList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[CSIDriver](#csidriver) array_ |  |


#### CSIDriverSpec



CSIDriverSpec defines the desired state of CSIDriver

_Appears in:_
- [CSIDriver](#csidriver)

| Field | Description |
| --- | --- |
| `image` _string_ | Image is the name of the csi driver image |
| `imagePullPolicy` _string_ | ImagePullPolicy is the image pull policy of the csi driver |
| `storageClassName` _string_ | StorageClassName is the name of the storage class |
| `volumeBindingMode` _string_ | VolumeBindingMode is the volume binding mode of the storage class |
| `sidecar` _[CSISidecar](#csisidecar)_ | Sidecar is the configuration for the CSI sidecar container nolint: lll |
| `clusters` _[VineyardClusters](#vineyardclusters) array_ | Clusters are the list of vineyard clusters |
| `enableToleration` _boolean_ | EnableToleration is the flag to enable toleration for the csi driver |
| `enableVerboseLog` _boolean_ | EnableVerboseLog is the flag to enable verbose log for the csi driver |




#### CSISidecar



CSISidecar holds the configuration for the CSI sidecar container

_Appears in:_
- [CSIDriverSpec](#csidriverspec)

| Field | Description |
| --- | --- |
| `provisionerImage` _string_ | ProvisionerImage is the image of the provisioner sidecar |
| `attacherImage` _string_ | AttacherImage is the image of the attacher sidecar |
| `nodeRegistrarImage` _string_ | NodeRegistrarImage is the image of the node registrar sidecar |
| `livenessProbeImage` _string_ | LivenessProbeImage is the image of the liveness probe sidecar |
| `imagePullPolicy` _string_ | ImagePullPolicy is the image pull policy of all sidecar containers |
| `enableTopology` _boolean_ | EnableTopology is the flag to enable topology for the csi driver |


#### GlobalObject



GlobalObject describes a global object in vineyard, whose metadata will be stored in etcd.

_Appears in:_
- [GlobalObjectList](#globalobjectlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `GlobalObject`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[GlobalObjectSpec](#globalobjectspec)_ |  |


#### GlobalObjectList



GlobalObjectList contains a list of GlobalObject



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `GlobalObjectList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[GlobalObject](#globalobject) array_ |  |


#### GlobalObjectSpec



GlobalObjectSpec defines the desired state of GlobalObject

_Appears in:_
- [GlobalObject](#globalobject)

| Field | Description |
| --- | --- |
| `id` _string_ |  |
| `name` _string_ |  |
| `signature` _string_ |  |
| `typename` _string_ |  |
| `members` _string array_ |  |
| `metadata` _string_ | Refer to Kubernetes API documentation for fields of `metadata`. |




#### LocalObject



LocalObject describes a local object in vineyard, whose metadata will only be stored in local vineyard.

_Appears in:_
- [LocalObjectList](#localobjectlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `LocalObject`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[LocalObjectSpec](#localobjectspec)_ |  |


#### LocalObjectList



LocalObjectList contains a list of LocalObject



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `LocalObjectList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[LocalObject](#localobject) array_ |  |


#### LocalObjectSpec



LocalObjectSpec defines the desired state of LocalObject

_Appears in:_
- [LocalObject](#localobject)

| Field | Description |
| --- | --- |
| `id` _string_ |  |
| `name` _string_ |  |
| `signature` _string_ |  |
| `typename` _string_ |  |
| `instance_id` _integer_ |  |
| `hostname` _string_ |  |
| `metadata` _string_ | Refer to Kubernetes API documentation for fields of `metadata`. |




#### MetricConfig



MetricConfig holds the configuration about metric container

_Appears in:_
- [SidecarSpec](#sidecarspec)
- [VineyarddSpec](#vineyarddspec)

| Field | Description |
| --- | --- |
| `enable` _boolean_ | Enable metrics |
| `image` _string_ | represent the metric's image |
| `imagePullPolicy` _string_ | the policy about pulling image |


#### Operation



Operation describes an operation between workloads, such as assembly and repartition. 
 As for the `assembly` operation, there are several kinds of computing engines, some may not support the stream data, so we need to insert an `assembly` operation to assemble the stream data into a batch data, so that the next computing engines can process the data. 
 As for the `repartition` operation, the vineyard has integrated with the distributed computing engines, such as Dask. If you want to repartition the data to adapt the dask workers, then the `repartition` operation is essential for such scenario.

_Appears in:_
- [OperationList](#operationlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `Operation`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[OperationSpec](#operationspec)_ |  |


#### OperationList



OperationList contains a list of Operation



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `OperationList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[Operation](#operation) array_ |  |


#### OperationSpec



OperationSpec defines the desired state of Operation

_Appears in:_
- [Operation](#operation)

| Field | Description |
| --- | --- |
| `name` _string_ | the name of vineyard pluggable drivers, including assembly and repartition. |
| `type` _string_ | the type of object, including local and distributed. |
| `require` _string_ | the required job's name of the operation |
| `target` _string_ | the target job's name of the operation |
| `timeoutSeconds` _integer_ | TimeoutSeconds is the timeout of the operation. |




#### PluginImageConfig



PluginImageConfig holds all image configuration about pluggable drivers(backup, recover, local assembly, distributed assembly, repartition)

_Appears in:_
- [VineyarddSpec](#vineyarddspec)

| Field | Description |
| --- | --- |
| `backupImage` _string_ | the image of backup operation |
| `recoverImage` _string_ | the image of recover operation |
| `daskRepartitionImage` _string_ | the image of dask repartition operation |
| `localAssemblyImage` _string_ | the image of local assembly operation |
| `distributedAssemblyImage` _string_ | the image of distributed assembly operation |


#### Recover



Recover describes a recover operation of vineyard objects, which is used to recover a specific backup operation.

_Appears in:_
- [RecoverList](#recoverlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `Recover`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[RecoverSpec](#recoverspec)_ |  |


#### RecoverList



RecoverList contains a list of Recover



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `RecoverList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[Recover](#recover) array_ |  |


#### RecoverSpec



RecoverSpec defines the desired state of Recover

_Appears in:_
- [Recover](#recover)

| Field | Description |
| --- | --- |
| `backupName` _string_ | the name of backup |
| `backupNamespace` _string_ | the namespace of backup |




#### ServiceConfig



ServiceConfig holds all service configuration about vineyardd

_Appears in:_
- [SidecarSpec](#sidecarspec)
- [VineyarddSpec](#vineyarddspec)

| Field | Description |
| --- | --- |
| `type` _string_ | service type |
| `port` _integer_ | service port |


#### Sidecar



Sidecar is used for configuring and managing the vineyard sidecar container.

_Appears in:_
- [SidecarList](#sidecarlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `Sidecar`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[SidecarSpec](#sidecarspec)_ |  |


#### SidecarList



SidecarList contains a list of Sidecar



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `SidecarList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[Sidecar](#sidecar) array_ |  |


#### SidecarSpec



SidecarSpec defines the desired state of Sidecar

_Appears in:_
- [Sidecar](#sidecar)

| Field | Description |
| --- | --- |
| `selector` _string_ | the selector of pod |
| `replicas` _integer_ | the replicas of workload |
| `etcdReplicas` _integer_ | EtcdReplicas describe the etcd replicas |
| `vineyard` _[VineyardConfig](#vineyardconfig)_ | vineyard container configuration nolint: lll |
| `metric` _[MetricConfig](#metricconfig)_ | metric container configuration |
| `volume` _[VolumeConfig](#volumeconfig)_ | metric configurations |
| `service` _[ServiceConfig](#serviceconfig)_ | rpc service configuration |
| `securityContext` _[SecurityContext](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#securitycontext-v1-core)_ | SecurityContext holds the security context settings for the vineyardd container. |
| `volumes` _[Volume](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#volume-v1-core) array_ | Volumes is the list of Kubernetes volumes that can be mounted by the vineyard container. |
| `volumeMounts` _[VolumeMount](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#volumemount-v1-core) array_ | VolumeMounts specifies the volumes listed in ".spec.volumes" to mount into the vineyard container. |




#### SpillConfig



SpillConfig holds all configuration about spilling

_Appears in:_
- [VineyardConfig](#vineyardconfig)

| Field | Description |
| --- | --- |
| `name` _string_ | the name of the spill config |
| `path` _string_ | the path of spilling |
| `spillLowerRate` _string_ | low watermark of spilling memory |
| `spillUpperRate` _string_ | high watermark of triggering spilling |
| `persistentVolumeSpec` _[PersistentVolumeSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#persistentvolumespec-v1-core)_ | the PersistentVolumeSpec of the spilling PV |
| `persistentVolumeClaimSpec` _[PersistentVolumeClaimSpec](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#persistentvolumeclaimspec-v1-core)_ | the PersistentVolumeClaimSpec of the spill file |


#### VineyardClusters



VineyardClusters contains the list of vineyard clusters

_Appears in:_
- [CSIDriverSpec](#csidriverspec)

| Field | Description |
| --- | --- |
| `namespace` _string_ | Namespace is the namespace of the vineyard cluster |
| `name` _string_ | Name is the name of the vineyard deployment |


#### VineyardConfig



VineyardConfig holds all configuration about vineyard container

_Appears in:_
- [SidecarSpec](#sidecarspec)
- [VineyarddSpec](#vineyarddspec)

| Field | Description |
| --- | --- |
| `image` _string_ | represent the vineyardd's image |
| `imagePullPolicy` _string_ | the policy about pulling image |
| `syncCRDs` _boolean_ | synchronize CRDs when persisting objects |
| `socket` _string_ | The directory on host for the IPC socket file. The UNIX-domain socket will be placed as `${Socket}/vineyard.sock`. |
| `size` _string_ | shared memory size for vineyardd |
| `reserveMemory` _boolean_ | reserve the shared memory for vineyardd |
| `streamThreshold` _integer_ | memory threshold of streams (percentage of total memory) |
| `spill` _[SpillConfig](#spillconfig)_ | the configuration of spilling |
| `env` _[EnvVar](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#envvar-v1-core) array_ | vineyard environment configuration |
| `memory` _string_ | the memory resources of vineyard container |
| `cpu` _string_ | the cpu resources of vineyard container |


#### Vineyardd



Vineyardd is used to deploy a vineyard cluster on kubernetes, which can simplify the configurations of the vineyard binary, the external etcd cluster and the vineyard [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/). As vineyard is bound to a specific socket on the hostpath by default, the vineyard pod cannot be deployed on the same node. Before deploying vineyardd, you should know how many nodes are available for vineyard pod to deploy on and make sure the vineyardd pod number is less than the number of available nodes.

_Appears in:_
- [VineyarddList](#vineyarddlist)

| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `Vineyardd`
| `metadata` _[ObjectMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#objectmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `spec` _[VineyarddSpec](#vineyarddspec)_ |  |


#### VineyarddList



VineyarddList contains a list of Vineyardd



| Field | Description |
| --- | --- |
| `apiVersion` _string_ | `k8s.v6d.io/v1alpha1`
| `kind` _string_ | `VineyarddList`
| `metadata` _[ListMeta](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#listmeta-v1-meta)_ | Refer to Kubernetes API documentation for fields of `metadata`. |
| `items` _[Vineyardd](#vineyardd) array_ |  |


#### VineyarddSpec



VineyarddSpec holds all configuration about vineyardd

_Appears in:_
- [Vineyardd](#vineyardd)

| Field | Description |
| --- | --- |
| `replicas` _integer_ | Replicas is the number of vineyardd pods to deploy |
| `etcdReplicas` _integer_ | EtcdReplicas describe the etcd replicas |
| `service` _[ServiceConfig](#serviceconfig)_ | vineyardd's service |
| `vineyard` _[VineyardConfig](#vineyardconfig)_ | vineyard container configuration nolint: lll |
| `pluginImage` _[PluginImageConfig](#pluginimageconfig)_ | operation container configuration nolint: lll |
| `metric` _[MetricConfig](#metricconfig)_ | metric container configuration |
| `volume` _[VolumeConfig](#volumeconfig)_ | Volume configuration |
| `securityContext` _[SecurityContext](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#securitycontext-v1-core)_ | SecurityContext holds the security context settings for the vineyardd container. |
| `volumes` _[Volume](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#volume-v1-core) array_ | Volumes is the list of Kubernetes volumes that can be mounted by the vineyard deployment. |
| `volumeMounts` _[VolumeMount](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.25/#volumemount-v1-core) array_ | VolumeMounts specifies the volumes listed in ".spec.volumes" to mount into the vineyard deployment. |




#### VolumeConfig



VolumeConfig holds all configuration about persistent volume

_Appears in:_
- [SidecarSpec](#sidecarspec)
- [VineyarddSpec](#vineyarddspec)

| Field | Description |
| --- | --- |
| `pvcName` _string_ | the name of pvc |
| `mountPath` _string_ | the mount path of pv |


