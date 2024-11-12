# `vineyardctl`

vineyardctl is the command-line tool for interact with the Vineyard Operator.

## Synopsis

vineyardctl is the command-line tool for working with the
Vineyard Operator. It supports creating, deleting and checking
status of Vineyard Operator. It also supports managing the
vineyard relevant components such as vineyardd and pluggable
drivers.

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes
* [vineyardctl csi](#vineyardctl-csi)	 - Start the vineyard csi driver
* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes
* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes
* [vineyardctl get](#vineyardctl-get)	 - Get vineyard object, metadata, blob or cluster-info
* [vineyardctl inject](#vineyardctl-inject)	 - Inject the vineyard sidecar container into a workload
* [vineyardctl ls](#vineyardctl-ls)	 - List vineyard objects, metadatas or blobs
* [vineyardctl manager](#vineyardctl-manager)	 - Start the manager of vineyard operator
* [vineyardctl put](#vineyardctl-put)	 - Put basic data type into vineyard
* [vineyardctl schedule](#vineyardctl-schedule)	 - Schedule a workload or a workflow to existing vineyard cluster.

## Options

```
      --create-namespace    create the namespace if it does not exist, default false
  -j, --dump-usage          Dump usage in JSON
  -g, --gen-doc             Generate reference docs, e.g., "./cmd/README.md"
  -h, --help                help for vineyardctl
      --kubeconfig string   kubeconfig path for the kubernetes cluster (default "$HOME/.kube/config")
  -n, --namespace string    the namespace for operation (default "vineyard-system")
      --verbose             print verbose log, default false
  -v, --version             version for vineyardctl
      --wait                wait for the kubernetes resource to be ready, default true (default true)
```

## `vineyardctl create`

Create a vineyard jobs on kubernetes

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl create backup](#vineyardctl-create-backup)	 - Create a backup cr to backup the current vineyard cluster on kubernetes
* [vineyardctl create operation](#vineyardctl-create-operation)	 - Insert an operation in a workflow based on vineyard cluster
* [vineyardctl create recover](#vineyardctl-create-recover)	 - Create a recover cr to recover the current vineyard cluster on kubernetes

### Examples

```shell
  # create the backup job on kubernetes
  vineyardctl create backup --vineyardd-name vineyardd-sample --vineyardd-namespace vineyard-system

  # create the recover job on kubernetes
  vineyardctl create recover --backup-name vineyardd-sample -n vineyard-system

  # create the operation job on kubernetes
  vineyardctl create operation --name assembly \
    --type local \
    --require job1 \
    --target job2 \
    --timeoutSeconds 600
```

### Options

```
  -h, --help   help for create
```

## `vineyardctl create backup`

Create a backup cr to backup the current vineyard cluster on kubernetes

### Synopsis

Backup the current vineyard cluster on kubernetes. You could
backup all objects of the current vineyard cluster quickly.
For persistent storage, you could specify the pv spec and pv
spec.

Notice, the command is used to create a backup cr for the
vineyard operator and you must deploy the vineyard operator
and vineyard cluster before using it.

```
vineyardctl create backup [flags]
```

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes

### Examples

```shell
  # create a backup cr for the vineyard cluster on kubernetes
  # you could define the pv and pvc spec from json string as follows
  vineyardctl create backup \
    --vineyardd-name vineyardd-sample \
    --vineyardd-namespace vineyard-system  \
    --limit 1000 \
    --path /var/vineyard/dump  \
    --pv-pvc-spec '{
      "pv-spec": {
        "capacity": {
          "storage": "1Gi"
        },
        "accessModes": [
          "ReadWriteOnce"
        ],
        "storageClassName": "manual",
        "hostPath": {
          "path": "/var/vineyard/dump"
        }
      },
      "pvc-spec": {
        "storageClassName": "manual",
        "accessModes": [
          "ReadWriteOnce"
        ],
        "resources": {
          "requests": {
          "storage": "1Gi"
          }
        }
      }
      }'

  # create a backup cr for the vineyard cluster on kubernetes
  # you could define the pv and pvc spec from yaml string as follows
  vineyardctl create backup \
    --vineyardd-name vineyardd-sample \
    --vineyardd-namespace vineyard-system  \
    --limit 1000 --path /var/vineyard/dump  \
    --pv-pvc-spec  \
    '
    pv-spec:
    capacity:
    storage: 1Gi
    accessModes:
    - ReadWriteOnce
    storageClassName: manual
    hostPath:
    path: "/var/vineyard/dump"
    pvc-spec:
    storageClassName: manual
    accessModes:
    - ReadWriteOnce
    resources:
    requests:
    storage: 1Gi
    '

  # create a backup cr for the vineyard cluster on kubernetes
  # you could define the pv and pvc spec from json file as follows
  # also you could use yaml file instead of json file
  cat pv-pvc.json | vineyardctl create backup \
    --vineyardd-name vineyardd-sample \
    --vineyardd-namespace vineyard-system  \
    --limit 1000 --path /var/vineyard/dump  \
    -
```

### Options

```
      --backup-name string           the name of backup job (default "vineyard-backup")
  -h, --help                         help for backup
      --objectIDs strings            the specific objects to be backed up
      --path string                  the path of the backup data
      --pv-pvc-spec string           the PersistentVolume and PersistentVolumeClaim of the backup data
      --vineyardd-name string        the name of vineyardd
      --vineyardd-namespace string   the namespace of vineyardd
```

## `vineyardctl create operation`

Insert an operation in a workflow based on vineyard cluster

### Synopsis

Insert an operation in a workflow based on vineyard cluster.
You could create a assembly or repartition operation in a
workflow. Usually, the operation should be created between
the workloads: job1 -> operation -> job2.

```
vineyardctl create operation [flags]
```

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes

### Examples

```shell
  # create a local assembly operation between job1 and job2
  vineyardctl create operation --name assembly \
    --type local \
    --require job1 \
    --target job2 \
    --timeoutSeconds 600

  # create a distributed assembly operation between job1 and job2
  vineyardctl create operation --name assembly \
    --type distributed \
    --require job1 \
    --target job2 \
    --timeoutSeconds 600

  # create a dask repartition operation between job1 and job2
  vineyardctl create operation --name repartition \
    --type dask \
    --require job1 \
    --target job2 \
    --timeoutSeconds 600
```

### Options

```
  -h, --help                 help for operation
      --kind string          the kind of operation, including "assembly" and "repartition"
      --name string          the name of operation
      --require string       the job that need an operation to be executed
      --target string        the job that need to be executed before this operation
      --timeoutSeconds int   the timeout seconds of operation (default 600)
      --type string          the type of operation: for assembly, it can be "local" or "distributed"; for repartition, it can be "dask"
```

## `vineyardctl create recover`

Create a recover cr to recover the current vineyard cluster on kubernetes

### Synopsis

Recover the current vineyard cluster on kubernetes. You could
recover all objects from a backup of vineyard cluster. Usually,
the recover crd should be created in the same namespace of
the backup job.

Notice, the command is used to create a recover cr for the
vineyard operator and you must deploy the vineyard operator
and vineyard cluster before using it.

```
vineyardctl create recover [flags]
```

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes

### Examples

```shell
  # create a recover cr for a backup job in the same namespace
  vineyardctl create recover --backup-name vineyardd-sample -n vineyard-system
```

### Options

```
      --backup-name string    the name of backup job (default "vineyard-backup")
  -h, --help                  help for recover
      --recover-name string   the name of recover job (default "vineyard-recover")
```

## `vineyardctl csi`

Start the vineyard csi driver

```
vineyardctl csi [flags]
```

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.

### Examples

```shell
  # start the csi with the specific endpoint and node id
  vineyardctl csi --endpoint=unix:///csi/csi.sock --nodeid=csinode1
```

### Options

```
  -f, --endpoint string          the endpoint of vineyard csi driver
  -h, --help                     help for csi
      --nodeid string            the node id of vineyard csi driver
      --state-file-path string   the path of state file (default "/csi/state")
```

## `vineyardctl delete`

Delete the vineyard components from kubernetes

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl delete backup](#vineyardctl-delete-backup)	 - Delete the backup job on kubernetes
* [vineyardctl delete csidriver](#vineyardctl-delete-csidriver)	 - Delete the vineyard csi driver on kubernetes
* [vineyardctl delete operation](#vineyardctl-delete-operation)	 - Delete the operation from kubernetes
* [vineyardctl delete operator](#vineyardctl-delete-operator)	 - Delete the vineyard operator from kubernetes
* [vineyardctl delete recover](#vineyardctl-delete-recover)	 - Delete the recover job from kubernetes
* [vineyardctl delete vineyard-cluster](#vineyardctl-delete-vineyard-cluster)	 - Delete the vineyard cluster from kubernetes
* [vineyardctl delete vineyard-deployment](#vineyardctl-delete-vineyard-deployment)	 - delete vineyard-deployment will delete the vineyard deployment without vineyard operator
* [vineyardctl delete vineyardd](#vineyardctl-delete-vineyardd)	 - Delete the vineyardd cluster from kubernetes

### Examples

```shell
  # delete the default vineyard cluster on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete

  # delete the default vineyard operator on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete operator

  # delete the default vineyardd on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyardd
```

### Options

```
  -h, --help   help for delete
```

## `vineyardctl delete backup`

Delete the backup job on kubernetes

### Synopsis

Delete the backup job on kubernetes.

```
vineyardctl delete backup [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default backup job
  vineyardctl delete backup
```

### Options

```
      --backup-name string   the name of backup job (default "vineyard-backup")
  -h, --help                 help for backup
```

## `vineyardctl delete csidriver`

Delete the vineyard csi driver on kubernetes

```
vineyardctl delete csidriver [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the csi driver named "csidriver-test"
  vineyardctl delete csidriver --name csidriver-test
```

### Options

```
  -h, --help          help for csidriver
      --name string   The name of the csi driver cr. (default "csidriver-sample")
```

## `vineyardctl delete operation`

Delete the operation from kubernetes

```
vineyardctl delete operation [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the operation named "assembly-test" in the "vineyard-system" namespace
  vineyardctl delete operation --name assembly-test
```

### Options

```
  -h, --help          help for operation
      --name string   the name of operation
```

## `vineyardctl delete operator`

Delete the vineyard operator from kubernetes

```
vineyardctl delete operator [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default vineyard operator in the vineyard-system namespace
  vineyardctl delete operator
  
  # delete the vineyard operator in a specific namespace
  vineyardctl delete operator -n <namespace>
```

### Options

```
  -h, --help   help for operator
```

## `vineyardctl delete recover`

Delete the recover job from kubernetes

```
vineyardctl delete recover [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default recover job on kubernetes
  vineyardctl delete recover
```

### Options

```
  -h, --help                  help for recover
      --recover-name string   the name of recover job (default "vineyard-recover")
```

## `vineyardctl delete vineyard-cluster`

Delete the vineyard cluster from kubernetes

```
vineyardctl delete vineyard-cluster [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default vineyard cluster on kubernetes
  vineyardctl delete vineyard-cluster
```

### Options

```
  -h, --help   help for vineyard-cluster
```

## `vineyardctl delete vineyard-deployment`

delete vineyard-deployment will delete the vineyard deployment without vineyard operator

```
vineyardctl delete vineyard-deployment [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default vineyard deployment in the vineyard-system namespace
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment

  # delete the vineyard deployment with specific name in the vineyard-system namespace
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete vineyard-deployment \
    --name vineyardd-0
```

### Options

```
  -h, --help          help for vineyard-deployment
      --name string   the name of vineyardd (default "vineyardd-sample")
```

## `vineyardctl delete vineyardd`

Delete the vineyardd cluster from kubernetes

```
vineyardctl delete vineyardd [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default vineyardd cluster(vineyardd-sample) in the default namespace
  vineyardctl delete vineyardd

  # delete the specific vineyardd cluster in the vineyard-system namespace
  vineyardctl -n vineyard-system delete vineyardd --name vineyardd-test
```

### Options

```
  -h, --help          help for vineyardd
      --name string   the name of vineyardd (default "vineyardd-sample")
```

## `vineyardctl deploy`

Deploy the vineyard components on kubernetes

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl deploy backup-job](#vineyardctl-deploy-backup-job)	 - Deploy a backup job of vineyard cluster on kubernetes
* [vineyardctl deploy csidriver](#vineyardctl-deploy-csidriver)	 - Deploy the vineyard csi driver on kubernetes
* [vineyardctl deploy operator](#vineyardctl-deploy-operator)	 - Deploy the vineyard operator on kubernetes
* [vineyardctl deploy recover-job](#vineyardctl-deploy-recover-job)	 - Deploy a recover job to recover a backup of current vineyard cluster on kubernetes
* [vineyardctl deploy vineyard-cluster](#vineyardctl-deploy-vineyard-cluster)	 - Deploy the vineyard cluster from kubernetes
* [vineyardctl deploy vineyard-deployment](#vineyardctl-deploy-vineyard-deployment)	 - DeployVineyardDeployment builds and deploy the yaml file of vineyardd without vineyard operator
* [vineyardctl deploy vineyardd](#vineyardctl-deploy-vineyardd)	 - Deploy the vineyardd on kubernetes

### Examples

```shell
  # deploy the default vineyard cluster on kubernetes
  vineyardctl --kubeconfig $HOME/.kube/config deploy vineyard-cluster

  # deploy the vineyard operator on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy operator

  # deploy the vineyardd on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd
  
  # deploy the vineyard csi driver on kubernetes
  vineyardctl deploy csidriver --name vineyard-csi-sample \
    --clusters vineyard-system/vineyardd-sample,default/vineyardd-sample
```

### Options

```
  -h, --help   help for deploy
```

## `vineyardctl deploy backup-job`

Deploy a backup job of vineyard cluster on kubernetes

### Synopsis

Deploy the backup job for the vineyard cluster on kubernetes,
which will backup all objects of the current vineyard cluster
quickly. For persistent storage, you could specify the pv spec
and pv spec and the related pv and pvc will be created automatically.
Also, you could also specify the existing pv and pvc name to use

```
vineyardctl deploy backup-job [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy a backup job for all vineyard objects of the vineyard
  # cluster on kubernetes and you could define the pv and pvc
  # spec from json string as follows
  vineyardctl deploy backup-job \
    --vineyard-deployment-name vineyardd-sample \
    --vineyard-deployment-namespace vineyard-system  \
    --path /var/vineyard/dump  \
    --pv-pvc-spec '{
      "pv-spec": {
        "capacity": {
          "storage": "1Gi"
        },
        "accessModes": [
          "ReadWriteOnce"
        ],
        "storageClassName": "manual",
        "hostPath": {
          "path": "/var/vineyard/dump"
        }
      },
      "pvc-spec": {
        "storageClassName": "manual",
        "accessModes": [
          "ReadWriteOnce"
        ],
        "resources": {
          "requests": {
          "storage": "1Gi"
          }
        }
      }
      }'

  # deploy a backup job for the vineyard cluster on kubernetes
  # you could define the pv and pvc spec from yaml string as follows
  vineyardctl deploy backup-job \
    --vineyard-deployment-name vineyardd-sample \
    --vineyard-deployment-namespace vineyard-system  \
    --path /var/vineyard/dump  \
    --pv-pvc-spec  \
    '
    pv-spec:
    capacity:
      storage: 1Gi
    accessModes:
    - ReadWriteOnce
    storageClassName: manual
    hostPath:
      path: "/var/vineyard/dump"
    pvc-spec:
    storageClassName: manual
    accessModes:
    - ReadWriteOnce
    resources:
      requests:
      storage: 1Gi
    '

  # deploy a backup job for specific vineyard objects of the vineyard
  # cluster on kubernetes.
  cat pv-pvc.json | vineyardctl deploy backup-job \
    --vineyard-deployment-name vineyardd-sample \
    --vineyard-deployment-namespace vineyard-system  \
    --objectIDs "o000018d29207fd01,o000018d80d264010"  \
    --path /var/vineyard/dump
    
  # Assume you have already deployed a pvc named "pvc-sample", you
  # could use them as the backend storage for the backup job as follows
  vineyardctl deploy backup-job \
    --vineyard-deployment-name vineyardd-sample \
    --vineyard-deployment-namespace vineyard-system  \
    --path /var/vineyard/dump  \
    --pvc-name pvc-sample
  
  # The namespace to deploy the backup and recover job must be the same
  # as the vineyard cluster namespace.
  # Assume the vineyard cluster is deployed in the namespace "test", you
  # could deploy the backup job as follows
  vineyardctl deploy backup-job \
    --vineyard-deployment-name vineyardd-sample \
    --vineyard-deployment-namespace test  \
    --namespace test  \
    --path /var/vineyard/dump  \
    --pvc-name pvc-sample
```

### Options

```
      --backup-name string                     the name of backup job (default "vineyard-backup")
  -h, --help                                   help for backup-job
      --objectIDs strings                      the specific objects to be backed up
      --path string                            the path of the backup data
      --pv-pvc-spec string                     the PersistentVolume and PersistentVolumeClaim of the backup data
      --pvc-name string                        the name of an existing PersistentVolumeClaim
      --vineyard-deployment-name string        the name of vineyard deployment
      --vineyard-deployment-namespace string   the namespace of vineyard deployment
```

## `vineyardctl deploy csidriver`

Deploy the vineyard csi driver on kubernetes

### Synopsis

Deploy the Vineyard CSI Driver on kubernetes. 
The CR is a cluster-scoped resource, and can only be created once.

```
vineyardctl deploy csidriver [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy the Vineyard CSI Driver named vineyard-csi-sample on kubernetes
  # Notice, the clusters are built as {vineyard-deployment-namespace}/{vineyard-deployment-name}
  # and sperated by comma, e.g. vineyard-system/vineyardd-sample, default/vineyardd-sample
  # They must be created before deploying the Vineyard CSI Driver.
  vineyardctl deploy csidriver --name vineyard-csi-sample \
    --clusters vineyard-system/vineyardd-sample,default/vineyardd-sample
```

### Options

```
      --attacherImage string             The image of csi attacher. (default "registry.k8s.io/sig-storage/csi-attacher:v4.0.0")
      --clusters strings                 The list of vineyard clusters.
      --enableToleration                 Enable toleration for vineyard csi driver.
  -h, --help                             help for csidriver
  -i, --image string                     The image of vineyard csi driver. (default "vineyardcloudnative/vineyard-operator")
      --imagePullPolicy string           The image pull policy of vineyard csi driver. (default "IfNotPresent")
      --livenessProbeImage string        The image of livenessProbe. (default "registry.k8s.io/sig-storage/livenessprobe:v2.8.0")
      --name string                      The name of the csi driver cr. (default "csidriver-sample")
      --nodeRegistrarImage string        The image of csi nodeRegistrar. (default "registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.6.0")
      --provisionerImage string          The image of csi provisioner. (default "registry.k8s.io/sig-storage/csi-provisioner:v3.3.0")
      --sidecar.enableTopology           Enable topology for the csi driver.
      --sidecar.imagePullPolicy string   The image pull policy of all sidecar containers. (default "Always")
  -s, --storageClassName string          The name of storage class. (default "vineyard-csi")
  -m, --volumeBindingMode string         The volume binding mode of the storage class. (default "WaitForFirstConsumer")
```

## `vineyardctl deploy operator`

Deploy the vineyard operator on kubernetes

### Synopsis

Deploy the vineyard operator on kubernetes.

```
vineyardctl deploy operator [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy the vineyard operator on the 'vineyard-system' namespace
  vineyardctl deploy operator
  
  # deploy the vineyard operator on the existing namespace
  vineyardctl deploy operator -n my-custom-namespace
  
  # deploy the vineyard operator on the new namespace
  vineyardctl deploy operator -n a-new-namespace-name --create-namespace
```

### Options

```
  -h, --help   help for operator
```

## `vineyardctl deploy recover-job`

Deploy a recover job to recover a backup of current vineyard cluster on kubernetes

### Synopsis

Deploy the recover job for vineyard cluster on kubernetes, which
will recover all objects from a backup of vineyard cluster. Usually,
the recover job should be created in the same namespace of
the backup job.

```
vineyardctl deploy recover-job [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # Deploy a recover job for the vineyard deployment in the same namespace.
  # After the recover job finished, the command will create a kubernetes
  # configmap named [recover-name]+"-mapping-table" that contains the
  # mapping table from the old vineyard objects to the new ones.
  #
  # If you create the recover job as follows, you can get the mapping table via
  # "kubectl get configmap vineyard-recover-mapping-table -n vineyard-system -o yaml"
  # the left column is the old object id, and the right column is the new object id.
  vineyardctl deploy recover-job \
  --vineyard-deployment-name vineyardd-sample \
  --vineyard-deployment-namespace vineyard-system  \
  --recover-path /var/vineyard/dump \
  --pvc-name vineyard-backup
```

### Options

```
  -h, --help                                   help for recover-job
      --pvc-name string                        the name of an existing PersistentVolumeClaim
      --recover-name string                    the name of recover job (default "vineyard-recover")
      --recover-path string                    the path of recover job
      --vineyard-deployment-name string        the name of vineyard deployment
      --vineyard-deployment-namespace string   the namespace of vineyard deployment
```

## `vineyardctl deploy vineyard-cluster`

Deploy the vineyard cluster from kubernetes

```
vineyardctl deploy vineyard-cluster [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy the default vineyard cluster on kubernetes
  vineyardctl deploy vineyard-cluster
```

### Options

```
  -h, --help   help for vineyard-cluster
```

## `vineyardctl deploy vineyard-deployment`

DeployVineyardDeployment builds and deploy the yaml file of vineyardd without vineyard operator

### Synopsis

Builds and deploy the yaml file of vineyardd the vineyardd
without vineyard operator. You could deploy a customized
vineyardd from stdin or file.

```
vineyardctl deploy vineyard-deployment [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy the default vineyard deployment on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config \
  deploy vineyard-deployment

  # deploy the vineyard deployment with customized image
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config \
  deploy vineyard-deployment --image vineyardcloudnative/vineyardd:v0.12.2
```

### Options

```
      --etcd.replicas int                             the number of etcd replicas in a vineyard cluster (default 1)
  -f, --file string                                   the path of vineyardd
  -h, --help                                          help for vineyard-deployment
      --name string                                   the name of vineyardd (default "vineyardd-sample")
      --owner-references string                       The owner reference of all vineyard deployment resources
      --pluginImage.backupImage string                the backup image of vineyardd (default "ghcr.io/v6d-io/v6d/backup-job")
      --pluginImage.daskRepartitionImage string       the dask repartition image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/dask-repartition")
      --pluginImage.distributedAssemblyImage string   the distributed image of vineyard workflow (default "ghcr.io/v6d-io/v6d/distributed-assembly")
      --pluginImage.localAssemblyImage string         the local assembly image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/local-assembly")
      --pluginImage.recoverImage string               the recover image of vineyardd (default "ghcr.io/v6d-io/v6d/recover-job")
      --replicas int                                  the number of vineyardd replicas (default 3)
      --securityContext string                        the json string of security context of vineyardd
      --vineyardd.cpu string                          the cpu requests and limits of vineyard container
      --vineyardd.envs strings                        The environment variables of vineyardd
      --vineyardd.image string                        the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --vineyardd.imagePullPolicy string              the imagePullPolicy of vineyardd (default "IfNotPresent")
      --vineyardd.memory string                       the memory requests and limits of vineyard container
      --vineyardd.metric.enable                       enable metrics of vineyardd
      --vineyardd.metric.image string                 the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --vineyardd.metric.imagePullPolicy string       the imagePullPolicy of the metric image (default "IfNotPresent")
      --vineyardd.reserve_memory                      Reserving enough physical memory pages for vineyardd
      --vineyardd.service.port int                    the service port of vineyard service (default 9600)
      --vineyardd.service.type string                 the service type of vineyard service (default "ClusterIP")
      --vineyardd.size string                         The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki. Defaults "", means not limited
      --vineyardd.socket string                       The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --vineyardd.spill.config string                 If you want to enable the spill mechanism, please set the name of spill config
      --vineyardd.spill.path string                   The path of spill config
      --vineyardd.spill.pv-pvc-spec string            the json string of the persistent volume and persistent volume claim
      --vineyardd.spill.spillLowerRate string         The low watermark of spilling memory (default "0.3")
      --vineyardd.spill.spillUpperRate string         The high watermark of spilling memory (default "0.8")
      --vineyardd.streamThreshold int                 memory threshold of streams (percentage of total memory) (default 80)
      --vineyardd.syncCRDs                            enable metrics of vineyardd (default true)
      --vineyardd.volume.mountPath string             Set the mount path for the pvc
      --vineyardd.volume.pvcname string               Set the pvc name for storing the vineyard objects persistently
      --volume string                                 the json string of vineyardd volume
      --volumeMount string                            the json string of vineyardd volume mount
```

## `vineyardctl deploy vineyardd`

Deploy the vineyardd on kubernetes

### Synopsis

Deploy the vineyardd on kubernetes. You could deploy a
customized vineyardd from stdin or file.

```
vineyardctl deploy vineyardd [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # deploy the default vineyard on kubernetes
  # wait for the vineyardd to be ready(default option)
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd

  # not to wait for the vineyardd to be ready
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
    --wait=false

  # deploy the vineyardd from a yaml file
  vineyardctl --kubeconfig $HOME/.kube/config deploy vineyardd --file vineyardd.yaml

  # deploy the vineyardd with customized image
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
    --image vineyardd:v0.12.2

  # deploy the vineyardd with spill mechanism on persistent storage from json string
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
    --vineyardd.spill.config spill-path \
    --vineyardd.spill.path /var/vineyard/spill \
    --vineyardd.spill.pv-pvc-spec '{
      "pv-spec": {
        "capacity": {
          "storage": "1Gi"
        },
        "accessModes": [
          "ReadWriteOnce"
        ],
        "storageClassName": "manual",
        "hostPath": {
          "path": "/var/vineyard/spill"
        }
      },
      "pvc-spec": {
        "storageClassName": "manual",
        "accessModes": [
          "ReadWriteOnce"
        ],
        "resources": {
          "requests": {
          "storage": "512Mi"
          }
        }
      }
    }'

  # deploy the vineyardd with spill mechanism on persistent storage from yaml string
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
    --vineyardd.spill.config spill-path \
    --vineyardd.spill.path /var/vineyard/spill \
    --vineyardd.spill.pv-pvc-spec \
    '
    pv-spec:
      capacity:
      storage: 1Gi
      accessModes:
      - ReadWriteOnce
      storageClassName: manual
      hostPath:
      path: "/var/vineyard/spill"
    pvc-spec:
      storageClassName: manual
      accessModes:
      - ReadWriteOnce
      resources:
      requests:
        storage: 512Mi
    '

# deploy the vineyardd with spill mechanism on persistent storage from json file
  # also you could use the yaml file
  cat pv-pvc.json | vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd \
    --vineyardd.spill.config spill-path \
    --vineyardd.spill.path /var/vineyard/spill \
    -
```

### Options

```
      --etcd.replicas int                             the number of etcd replicas in a vineyard cluster (default 1)
  -f, --file string                                   the path of vineyardd
  -h, --help                                          help for vineyardd
      --name string                                   the name of vineyardd (default "vineyardd-sample")
      --pluginImage.backupImage string                the backup image of vineyardd (default "ghcr.io/v6d-io/v6d/backup-job")
      --pluginImage.daskRepartitionImage string       the dask repartition image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/dask-repartition")
      --pluginImage.distributedAssemblyImage string   the distributed image of vineyard workflow (default "ghcr.io/v6d-io/v6d/distributed-assembly")
      --pluginImage.localAssemblyImage string         the local assembly image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/local-assembly")
      --pluginImage.recoverImage string               the recover image of vineyardd (default "ghcr.io/v6d-io/v6d/recover-job")
      --replicas int                                  the number of vineyardd replicas (default 3)
      --securityContext string                        the json string of security context of vineyardd
      --vineyardd.cpu string                          the cpu requests and limits of vineyard container
      --vineyardd.envs strings                        The environment variables of vineyardd
      --vineyardd.image string                        the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --vineyardd.imagePullPolicy string              the imagePullPolicy of vineyardd (default "IfNotPresent")
      --vineyardd.memory string                       the memory requests and limits of vineyard container
      --vineyardd.metric.enable                       enable metrics of vineyardd
      --vineyardd.metric.image string                 the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --vineyardd.metric.imagePullPolicy string       the imagePullPolicy of the metric image (default "IfNotPresent")
      --vineyardd.reserve_memory                      Reserving enough physical memory pages for vineyardd
      --vineyardd.service.port int                    the service port of vineyard service (default 9600)
      --vineyardd.service.type string                 the service type of vineyard service (default "ClusterIP")
      --vineyardd.size string                         The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki. Defaults "", means not limited
      --vineyardd.socket string                       The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --vineyardd.spill.config string                 If you want to enable the spill mechanism, please set the name of spill config
      --vineyardd.spill.path string                   The path of spill config
      --vineyardd.spill.pv-pvc-spec string            the json string of the persistent volume and persistent volume claim
      --vineyardd.spill.spillLowerRate string         The low watermark of spilling memory (default "0.3")
      --vineyardd.spill.spillUpperRate string         The high watermark of spilling memory (default "0.8")
      --vineyardd.streamThreshold int                 memory threshold of streams (percentage of total memory) (default 80)
      --vineyardd.syncCRDs                            enable metrics of vineyardd (default true)
      --vineyardd.volume.mountPath string             Set the mount path for the pvc
      --vineyardd.volume.pvcname string               Set the pvc name for storing the vineyard objects persistently
      --volume string                                 the json string of vineyardd volume
      --volumeMount string                            the json string of vineyardd volume mount
```

## `vineyardctl get`

Get vineyard object, metadata, blob or cluster-info

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl get blob](#vineyardctl-get-blob)	 - Get vineyard blob
* [vineyardctl get cluster-info](#vineyardctl-get-cluster-info)	 - Get vineyard cluster info
* [vineyardctl get metadata](#vineyardctl-get-metadata)	 - Get vineyard metadata
* [vineyardctl get object](#vineyardctl-get-object)	 - Get vineyard object

### Examples

```shell
  # Connect the vineyardd deployment with IPC client
  # Get the cluster info and output as table
  vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system
```

### Options

```
  -h, --help   help for get
```

## `vineyardctl get blob`

Get vineyard blob

### Synopsis

Get vineyard blob and only support IPC socket.
If you don't specify the ipc socket every time, you can set it as the 
environment variable VINEYARD_IPC_SOCKET.

```
vineyardctl get blob [flags]
```

**SEE ALSO**

* [vineyardctl get](#vineyardctl-get)	 - Get vineyard object, metadata, blob or cluster-info

### Examples

```shell
  # Get vineyard blob with the given vineyard object_id and the ipc socket
  vineyardctl get blob --object_id xxxxxxxx --ipc-socket /var/run/vineyard.sock

  # Get vineyard blob with the given vineyard object_id and the ipc socket
  # and set the unsafe to be true
  vineyardctl get blob --object_id xxxxxxxx --unsafe --ipc-socket /var/run/vineyard.sock
  
  # If you set the environment variable VINEYARD_IPC_SOCKET
  # you can use the following command to get vineyard blob
  vineyardctl get blob --object_id xxxxxxxx
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for blob
      --ipc-socket string        vineyard IPC socket path
      --object_id string         The object id to get
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
      --syncRemote               If the target object is a remote object,code_remote=True will force a meta synchronization on the vineyard server.
      --unsafe                   unsafe means getting the blob even the blob is not sealed yet.Default is False.
```

## `vineyardctl get cluster-info`

Get vineyard cluster info

### Synopsis

Get vineyard cluster info, including
the instanceId, hostName, node name and so on.

```
vineyardctl get cluster-info [flags]
```

**SEE ALSO**

* [vineyardctl get](#vineyardctl-get)	 - Get vineyard object, metadata, blob or cluster-info

### Examples

```shell
  # Get the cluster info of vineyard deployment and output as table
  vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system

  # Get the cluster info of vineyard deployment and output as json
  vineyardctl get cluster-info --deployment-name vineyardd-sample -n vineyard-system -o json
  
  # Get the cluster info via IPC socket
  vineyardctl get cluster-info --ipc-socket /var/run/vineyard.sock
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for cluster-info
      --ipc-socket string        vineyard IPC socket path
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
```

## `vineyardctl get metadata`

Get vineyard metadata

### Synopsis

Get vineyard metadata and support IPC socket,
RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or 
VINEYARD_RPC_SOCKET.

```
vineyardctl get metadata [flags]
```

**SEE ALSO**

* [vineyardctl get](#vineyardctl-get)	 - Get vineyard object, metadata, blob or cluster-info

### Examples

```shell
  # Get vineyard metadata with the given vineyard object_id and the ipc socket
  vineyardctl get metadata --object_id xxxxxxxx --ipc-socket /var/run/vineyard.sock

  # Get vineyard metadata with the given vineyard object_id and the ipc socket
  # and set the syncRemote to be true
  vineyardctl get metadata --object_id xxxxxxxx --syncRemote --ipc-socket /var/run/vineyard.sock

  # Get vineyard metadata with the given vineyard object_id and the rpc socket
  vineyardctl get metadata --object_id xxxxxxxx --rpc-socket 127.0.0.1:9600
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for metadata
      --ipc-socket string        vineyard IPC socket path
      --object_id string         The object id to get
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
      --syncRemote               If the target object is a remote object,code_remote=True will force a meta synchronization on the vineyard server.
      --unsafe                   unsafe means getting the blob even the blob is not sealed yet.Default is False.
```

## `vineyardctl get object`

Get vineyard object

### Synopsis

Get vineyard object and support IPC socket,
RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or
VINEYARD_RPC_SOCKET.

```
vineyardctl get object [flags]
```

**SEE ALSO**

* [vineyardctl get](#vineyardctl-get)	 - Get vineyard object, metadata, blob or cluster-info

### Examples

```shell
  # Get vineyard object with the given vineyard object_id and the ipc socket
  vineyardctl get object --object_id xxxxxxxx --ipc-socket /var/run/vineyard.sock

  # Get vineyard object with the given vineyard object_id and the rpc socket
  vineyardctl get object --object_id xxxxxxxx --rpc-socket 127.0.0.1:9600
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for object
      --ipc-socket string        vineyard IPC socket path
      --object_id string         The object id to get
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
      --syncRemote               If the target object is a remote object,code_remote=True will force a meta synchronization on the vineyard server.
      --unsafe                   unsafe means getting the blob even the blob is not sealed yet.Default is False.
```

## `vineyardctl inject`

Inject the vineyard sidecar container into a workload

### Synopsis

Inject the vineyard sidecar container into a workload. You can
input a workload yaml or a workload json and then get the injected
workload and some etcd manifests from the output. The workload can
be a pod or a deployment or a statefulset, etc.

The output is a set of manifests that includes the injected workload,
the rpc service, the etcd service and the etcd cluster(e.g. several
pods and services). 

If you have a pod yaml: 

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: python
spec:
  containers:
  - name: python
    image: python:3.10
    command: ["python", "-c", "import time; time.sleep(100000)"]
```
Then, you can use the following command to inject the vineyard sidecar

$ vineyardctl inject -f pod.yaml

After running the command, the output is as follows:

```yaml
apiVersion: v1
kind: Pod
metadata:
  labels:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: etcd
    etcd_node: vineyard-sidecar-etcd-0
  name: vineyard-sidecar-etcd-0
  namespace: null
  ownerReferences: []
spec:
  containers:
  - command:
    - etcd
    - --name
    - vineyard-sidecar-etcd-0
    - --initial-advertise-peer-urls
    - http://vineyard-sidecar-etcd-0:2380
    - --advertise-client-urls
    - http://vineyard-sidecar-etcd-0:2379
    - --listen-peer-urls
    - http://0.0.0.0:2380
    - --listen-client-urls
    - http://0.0.0.0:2379
    - --initial-cluster
    - vineyard-sidecar-etcd-0=http://vineyard-sidecar-etcd-0:2380
    - --initial-cluster-state
    - new
    image: vineyardcloudnative/vineyardd:latest
    name: etcd
    ports:
    - containerPort: 2379
      name: client
      protocol: TCP
    - containerPort: 2380
      name: server
      protocol: TCP
  restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  labels:
    etcd_node: vineyard-sidecar-etcd-0
  name: vineyard-sidecar-etcd-0
  namespace: null
  ownerReferences: []
spec:
  ports:
  - name: client
    port: 2379
    protocol: TCP
    targetPort: 2379
  - name: server
    port: 2380
    protocol: TCP
    targetPort: 2380
  selector:
    app.vineyard.io/role: etcd
    etcd_node: vineyard-sidecar-etcd-0
---
apiVersion: v1
kind: Service
metadata:
  name: vineyard-sidecar-etcd-service
  namespace: null
  ownerReferences: []
spec:
  ports:
  - name: etcd-for-vineyard-port
    port: 2379
    protocol: TCP
    targetPort: 2379
  selector:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: etcd
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app.vineyard.io/name: vineyard-sidecar
  name: vineyard-sidecar-rpc
  namespace: null
  ownerReferences: []
spec:
  ports:
  - name: vineyard-rpc
    port: 9600
    protocol: TCP
  selector:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: vineyardd
  type: ClusterIP
---
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    app.vineyard.io/name: vineyard-sidecar
    app.vineyard.io/role: vineyardd
  name: python
  ownerReferences: []
spec:
  containers:
  - command:
    - python
    - -c
    - while [ ! -e /var/run/vineyard.sock ]; do sleep 1; done;import time; time.sleep(100000)
    env:
    - name: VINEYARD_IPC_SOCKET
      value: /var/run/vineyard.sock
    image: python:3.10
    name: python
    resources: {}
    volumeMounts:
    - mountPath: /var/run
      name: vineyard-socket
  - command:
    - /bin/bash
    - -c
    - |
      /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock --size \
      --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
    env:
    - name: VINEYARDD_UID
      value: null
    - name: VINEYARDD_NAME
      value: vineyard-sidecar
    - name: VINEYARDD_NAMESPACE
      value: null
    image: vineyardcloudnative/vineyardd:latest
    imagePullPolicy: IfNotPresent
    name: vineyard-sidecar
    ports:
    - containerPort: 9600
      name: vineyard-rpc
      protocol: TCP
    resources:
      limits: null
      requests: null
    securityContext: {}
    volumeMounts:
    - mountPath: /var/run
      name: vineyard-socket
  volumes:
  - emptyDir: {}
    name: vineyard-socket
status: {}
```

Next, we will introduce a simple example to show the injection with
the apply-resources flag.

Assume you have the following workload yaml:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
    # Notice, you must set the namespace here
  namespace: vineyard-job
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

Then, you can use the following command to inject the vineyard sidecar
which means that all resources will be created during the injection except
the workload itself. The workload should be created by users.

$ vineyardctl inject -f workload.yaml --apply-resources

After running the command, the main output(removed some unnecessary fields)
is as follows:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  name: nginx-deployment
  namespace: vineyard-job
spec:
  selector:
    matchLabels:
      app: nginx
template:
  metadata:
  labels:
    app: nginx
    # the default sidecar name is vineyard-sidecar
    app.vineyard.io/name: vineyard-sidecar
  spec:
    containers:
    - command: null
      image: nginx:1.14.2
      name: nginx
      ports:
      - containerPort: 80
      volumeMounts:
      - mountPath: /var/run
        name: vineyard-socket
    - command:
      - /bin/bash
      - -c
      - |
        /usr/local/bin/vineyardd --sync_crds true --socket /var/run/vineyard.sock \
        --stream_threshold 80 --etcd_cmd etcd --etcd_prefix /vineyard \
        --etcd_endpoint http://vineyard-sidecar-etcd-service:2379
      env:
      - name: VINEYARDD_UID
        value: null
      - name: VINEYARDD_NAME
        value: vineyard-sidecar
      - name: VINEYARDD_NAMESPACE
        value: vineyard-job
      image: vineyardcloudnative/vineyardd:latest
      imagePullPolicy: IfNotPresent
      name: vineyard-sidecar
      ports:
      - containerPort: 9600
        name: vineyard-rpc
        protocol: TCP
      volumeMounts:
      - mountPath: /var/run
        name: vineyard-socket
    volumes:
    - emptyDir: {}
      name: vineyard-socket
```

The sidecar template can be accessed from the following link:
https://github.com/v6d-io/v6d/blob/main/k8s/pkg/templates/sidecar/injection-template.yaml
also you can get some inspiration from the doc link:
https://v6d.io/notes/cloud-native/vineyard-operator.html#installing-vineyard-as-sidecar

```
vineyardctl inject [flags]
```

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl inject argo-workflow](#vineyardctl-inject-argo-workflow)	 - Inject the vineyard volumes into the argo workflow

### Examples

```shell
  # use json format to output the injected workload
  # notice that the output is a json string of all manifests
  # it looks like:
  # {
  #   "workload": "workload json string",
  #   "rpc_service": "rpc service json string",
  #   "etcd_service": "etcd service json string",
  #   "etcd_internal_service": [
  #     "etcd internal service json string 1",
  #     "etcd internal service json string 2",
  #     "etcd internal service json string 3"
  #   ],
  #   "etcd_pod": [
  #     "etcd pod json string 1",
  #     "etcd pod json string 2",
  #     "etcd pod json string 3"
  #   ]
  # }
  vineyardctl inject -f workload.yaml -o json

  # inject the default vineyard sidecar container into a workload
  # output all injected manifests and then deploy them
  vineyardctl inject -f workload.yaml | kubectl apply -f -

  # if you only want to get the injected workload yaml rather than
  # all manifests that includes the etcd cluster and the rpc service,
  # you can enable the apply-resources and then the manifests will be
  # created during the injection, finally you will get the injected
  # workload yaml
  vineyardctl inject -f workload.yaml --apply-resources
```

### Options

```
      --apply-resources                         Whether to apply the resources including the etcd cluster and the rpc service if you enable this flag, the etcd cluster and the rpc service will be created during the injection
      --etcd-replicas int                       The number of etcd replicas (default 1)
  -f, --file string                             The yaml of workload
  -h, --help                                    help for inject
      --name string                             The name of sidecar (default "vineyard-sidecar")
  -o, --output string                           The output format of the command, support yaml and json (default "yaml")
      --owner-references string                 The owner reference of all injectied resources
      --resource string                         The resource of workload
      --securityContext string                  the json string of security context of vineyard sidecar container
      --sidecar.cpu string                      the cpu requests and limits of vineyard container
      --sidecar.envs strings                    The environment variables of vineyardd
      --sidecar.image string                    the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --sidecar.imagePullPolicy string          the imagePullPolicy of vineyardd (default "IfNotPresent")
      --sidecar.memory string                   the memory requests and limits of vineyard container
      --sidecar.metric.enable                   enable metrics of vineyardd
      --sidecar.metric.image string             the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --sidecar.metric.imagePullPolicy string   the imagePullPolicy of the metric image (default "IfNotPresent")
      --sidecar.reserve_memory                  Reserving enough physical memory pages for vineyardd
      --sidecar.service.port int                the service port of vineyard service (default 9600)
      --sidecar.service.type string             the service type of vineyard service (default "ClusterIP")
      --sidecar.size string                     The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki. Defaults "", means not limited
      --sidecar.socket string                   The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --sidecar.spill.config string             If you want to enable the spill mechanism, please set the name of spill config
      --sidecar.spill.path string               The path of spill config
      --sidecar.spill.pv-pvc-spec string        the json string of the persistent volume and persistent volume claim
      --sidecar.spill.spillLowerRate string     The low watermark of spilling memory (default "0.3")
      --sidecar.spill.spillUpperRate string     The high watermark of spilling memory (default "0.8")
      --sidecar.streamThreshold int             memory threshold of streams (percentage of total memory) (default 80)
      --sidecar.syncCRDs                        enable metrics of vineyardd (default true)
      --sidecar.volume.mountPath string         Set the mount path for the pvc
      --sidecar.volume.pvcname string           Set the pvc name for storing the vineyard objects persistently
      --volume string                           the json string of vineyard sidecar container volume
      --volumeMount string                      the json string of vineyard sidecar container volume mount
```

## `vineyardctl inject argo-workflow`

Inject the vineyard volumes into the argo workflow

### Synopsis

Inject the vineyard volumes into the argo workflow DAGs. You can input
the workflow manifest file and the injected manifest file with 
vineyard volume will be output to the file with the suffix 
"_with_vineyard", such as "workflow_with_vineyard.yaml".

Suppose the workflow manifest named "workflow.yaml" is as follows:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - name: producer
    container:
      image: producer:latest
      command: [python]
      args: ["/producer.py"]
  - name: consumer
    container:
      image: consumer:latest
      command: [python]
      args: ["/consumer.py"]
  - name: dag
    dag:
      tasks:
      - name: producer
        template: producer
      - name: consumer
        template: consumer
        dependencies:
          - producer
```

Assume the 'producer' and 'consumer' task all need to use vineyard 
volume, you can inject the vineyard volume into the workflow manifest 
with the following command:

$ vineyardctl inject argo-workflow -f workflow.yaml \
      --templates="producer,consumer" \
  --vineyard-cluster="vineyard-system/vineyardd-sample" \
  --mount-path="/vineyard/data" \
  --dag="dag" \
  --tasks="producer,consumer" \
  --output-as-file

The injected manifest will be output to the file named "workflow_with_vineyard.yaml".

$ cat workflow_with_vineyard.yaml

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - name: producer
    container:
      image: producer:latest
      command: [python]
      args: ["/producer.py"]
      ################## Injected #################
      volumeMounts: 
      - name: vineyard-objects
        mountPath: /vineyard/data
      #############################################
    ######################## Injected #######################
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim: 
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
    #########################################################
    ############## Injected #############
    inputs:
      parameters:
      - {name: vineyard-objects-name}
    #####################################
  - name: consumer
    container:
      image: consumer:latest
      command: [python]
      args: ["/consumer.py"]
      ################## Injected #################
      volumeMounts: 
      - name: vineyard-objects
        mountPath: /vineyard/data
      #############################################
    ######################## Injected #######################
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim: 
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
    #########################################################
    ############## Injected #############
    inputs:
      parameters:
      - {name: vineyard-objects-name}
    #####################################
  - name: dag
    dag:
      tasks:
      - name: producer
        template: producer
        arguments:
          parameters:
          ################################# Injected ################################
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
          ###########################################################################
        dependencies:
          ########### Injected ##########
          - vineyard-objects
          ###############################
      - name: consumer
        template: consumer
        arguments:
          parameters:
          ################################# Injected ################################
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
          ###########################################################################
        dependencies:
          - producer
          ########### Injected ##########
          - vineyard-objects
          ###############################
      ########## Injected #########
      - name: vineyard-objects
        template: vineyard-objects
      #############################
############################# Injected ##########################
  - name: vineyard-objects
    resource:
      action: create
      setOwnerReference: true
      manifest: |
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
          name: '{{workflow.name}}-vineyard-objects-pvc'
        spec:
          accessModes:
          - ReadWriteMany
          resources:
            requests:
              storage: 1Mi
          storageClassName: vineyard-system.vineyardd-sample.csi
    outputs:
      parameters:
      - name: vineyard-objects-name
        valueFrom:
          jsonPath: '{.metadata.name}'  
##############################################################
```
Suppose your workflow YAML only has a single template as follows.
$ cat workflow.yaml

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - name: MLops
    inputs:
      parameters:
      - name: functions
    container:
      imagePullPolicy: IfNotPresent
      image: mlops-benchmark:latest
      command: [python]
      args: ["-m",  "{{inputs.parameters.functions}}"]
  - name: dag
    dag:
      tasks:
      - name: producer
        template: MLops
        arguments:
          parameters:
          - name: functions
            value: producer.py
      - name: consumer
        template: MLops
        dependencies:
          - producer
        arguments:
          parameters:
          - name: functions
            value: consumer.py
```
Suppose only the 'consumer' task need to use vineyard volume,
you can inject the vineyard volume into the workflow manifest
with the following command:
$ vineyardctl inject argo-workflow -f workflow.yaml \
     --templates="MLops" \
     --vineyard-cluster="vineyard-system/vineyardd-sample" \
     --mount-path="/vineyard/data" \
     --dag="dag" \
     --tasks="consumer"

Then the injected manifest will be as follows:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: mlops-
spec:
  entrypoint: dag
  templates:
  - container:
      args:
      - -m
      - '{{inputs.parameters.functions}}'
      command:
      - python
      image: mlops-benchmark:latest
      imagePullPolicy: IfNotPresent
      ############# Injected ############
      volumeMounts:
      - name: vineyard-objects
        mountPath: /vineyard/data
      ###################################
    inputs:
      parameters:
      - name: functions
      ############# Injected ############
      - name: vineyard-objects-name
      ###################################
    name: MLops
    ######################### Injected ########################
    volumes:
    - name: vineyard-objects
      persistentVolumeClaim:
        claimName: '{{inputs.parameters.vineyard-objects-name}}'
    ###########################################################
  - dag:
      tasks:
      - arguments:
          parameters:
          - name: functions
            value: producer.py
          ################################# Injected #################################
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
          ############################################################################
        dependencies:
        ##### Injected #####
        - vineyard-objects
        ####################
        name: producer
        template: MLops
      - arguments:
          parameters:
          - name: functions
            value: consumer.py
          ################################# Injected #################################
          - name: vineyard-objects-name
            value: '{{tasks.vineyard-objects.outputs.parameters.vineyard-objects-name}}'
          ############################################################################
        dependencies:
        - producer
        ##### Injected #####
        - vineyard-objects
        ####################
        name: consumer
        template: MLops
      ########### Injected ###########
      - name: vineyard-objects
        template: vineyard-objects
      ################################
    name: dag
  ################################# Injected #################################
  - name: vineyard-objects
    outputs:
      parameters:
      - name: vineyard-objects-name
        valueFrom:
          jsonPath: '{.metadata.name}'
    resource:
      action: create
      manifest: |-
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
          name: '{{workflow.name}}-vineyard-objects-pvc'
        spec:
          accessModes:
          - ReadWriteMany
          resources:
            requests:
              storage: 1Mi
          storageClassName: vineyard-system.vineyardd-sample.csi
      setOwnerReference: true
      ############################################################################
```

```
vineyardctl inject argo-workflow [flags]
```

**SEE ALSO**

* [vineyardctl inject](#vineyardctl-inject)	 - Inject the vineyard sidecar container into a workload

### Examples

```shell
  # Inject the vineyard volumes into the argo workflow
  $ vineyardctl inject argo-workflow -f workflow.yaml \
  --templates="preprocess-data,train-data" \
  --vineyard-cluster="vineyard-system/vineyardd-sample" \
  --mount-path="/vineyard/data" \
  --dag="dag" \
  --tasks="preprocess-data,train-data"
  
  # Suppose you only have a single template in the workflow
  # you could set only one template name in the --templates flag
  $ vineyardctl inject argo-workflow -f workflow.yaml \
--templates="mlops" \
    --vineyard-cluster="vineyard-system/vineyardd-sample" \
    --mount-path="/vineyard/data" \
    --dag="dag" \
    --tasks="preprocess-data,test-data"
```

### Options

```
      --dag string                The name of dag which will be injected with vineyard volumes
  -f, --file string               The file name of argo workflow
  -h, --help                      help for argo-workflow
      --mount-path string         The mount path of vineyard volumes
      --output-as-file            Whether to output the injected workflow as a file, default is falseThe output file name will add a suffix '_with_vineyard' to the original file name
      --tasks strings             The set of task names under the dag
  -t, --templates strings         The name of workflow template which will be injected with vineyard volumes
      --vineyard-cluster string   The name of vineyard cluster which the argo workflow will use
```

## `vineyardctl ls`

List vineyard objects, metadatas or blobs

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl ls blobs](#vineyardctl-ls-blobs)	 - List vineyard blobs
* [vineyardctl ls metadatas](#vineyardctl-ls-metadatas)	 - List vineyard metadatas
* [vineyardctl ls objects](#vineyardctl-ls-objects)	 - List vineyard objects

### Examples

```shell
  # Connect the vineyardd server with IPC client
  # List the vineyard objects no more than 10
  vineyardctl ls objects --limit 10 --ipc-socket /var/run/vineyard.sock

  # List the vineyard blobs no more than 10
  vineyardctl ls blobs --limit 10 --ipc-socket /var/run/vineyard.sock

  # List the vineyard objects with the specified pattern
  vineyardctl ls objects --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock

  # Connect the vineyardd server with RPC client
  # List the vineyard metadatas no more than 1000
  vineyardctl ls metadatas --rpc-socket 127.0.0.1:9600 --limit 1000

  # Connect the vineyard deployment with PRC client
  # List the vineyard objects no more than 1000
  vineyardctl ls objects --deployment-name vineyardd-sample -n vineyard-system
```

### Options

```
  -h, --help   help for ls
```

## `vineyardctl ls blobs`

List vineyard blobs

### Synopsis

List vineyard blobs and only support IPC socket.
If you don't specify the ipc socket every time, you can set it as the 
environment variable VINEYARD_IPC_SOCKET.

```
vineyardctl ls blobs [flags]
```

**SEE ALSO**

* [vineyardctl ls](#vineyardctl-ls)	 - List vineyard objects, metadatas or blobs

### Examples

```shell
  # List no more than 10 vineyard blobs
  vineyardctl ls blobs --limit 10 --ipc-socket /var/run/vineyard.sock

  # List no more than 1000 vineyard blobs
  vineyardctl ls blobs --ipc-socket /var/run/vineyard.sock --limit 1000
  
  # List vineyard blobs with the name matching
  vineyardctl ls blobs --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock
  
  # List vineyard blobs with the regex pattern
  vineyardctl ls blobs --pattern "*DataFrame*" --ipc-socket /var/run/vineyard.sock
  
  # If you set the environment variable VINEYARD_IPC_SOCKET
  # you can use the following command to list vineyard blobs
  vineyardctl ls blobs --limit 1000
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for blobs
      --ipc-socket string        vineyard IPC socket path
  -l, --limit int                maximum number of objects to return (default 5)
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
```

## `vineyardctl ls metadatas`

List vineyard metadatas

### Synopsis

List vineyard metadatas and support IPC socket,
RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or 
VINEYARD_RPC_SOCKET.

```
vineyardctl ls metadatas [flags]
```

**SEE ALSO**

* [vineyardctl ls](#vineyardctl-ls)	 - List vineyard objects, metadatas or blobs

### Examples

```shell
  # List no more than 10 vineyard metadatas
  vineyardctl ls metadatas --limit 10 --ipc-socket /var/run/vineyard.sock
  
  # List no more than 1000 vineyard metadatas
  vineyardctl ls metadatas --rpc-socket 127.0.0.1:9600 --limit 1000
  
  # List vineyard metadatas with the name matching the regex pattern
  vineyardctl ls metadatas --pattern "vineyard::Blob" --ipc-socket /var/run/vineyard.sock

  # List vineyard metadatas of the vineyard deployment
  vineyardctl ls metadatas --deployment-name vineyardd-sample -n vineyard-system --limit 1000
  
  # List vineyard metadatas sorted by the instance id
  vineyardctl ls metadatas --sorted-key instance_id --limit 1000 --ipc-socket /var/run/vineyard.sock

  # List vineyard metadatas sorted by the type and print the output as json format
  vineyardctl ls metadatas --sorted-key type --limit 1000 --format json --ipc-socket /var/run/vineyard.sock
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for metadatas
      --ipc-socket string        vineyard IPC socket path
  -l, --limit int                maximum number of objects to return (default 5)
  -p, --pattern string           string that will be matched against the object’s typenames (default "*")
      --port int                 the port of vineyard deployment (default 9600)
  -r, --regex                    regex pattern to match the object’s typenames
      --rpc-socket string        vineyard RPC socket path
  -k, --sorted-key string        key to sort the objects, support:
                                 - id: object id, the default value.
                                 - typename: object typename, e.g. tensor, dataframe, etc.
                                 - type: object type, e.g. global, local, etc.
                                 - instance_id: object instance id. (default "id")
```

## `vineyardctl ls objects`

List vineyard objects

### Synopsis

List vineyard objects and support IPC socket,
RPC socket and vineyard deployment. If you don't specify the ipc socket or rpc socket
every time, you can set it as the environment variable VINEYARD_IPC_SOCKET or
VINEYARD_RPC_SOCKET.

```
vineyardctl ls objects [flags]
```

**SEE ALSO**

* [vineyardctl ls](#vineyardctl-ls)	 - List vineyard objects, metadatas or blobs

### Examples

```shell
  # List no more than 10 vineyard objects
  vineyardctl ls objects --limit 10 --ipc-socket /var/run/vineyard.sock

  # List any vineyard objects and no more than 1000 objects
  vineyardctl ls objects --pattern "*" --ipc-socket /var/run/vineyard.sock --limit 1000

  # List vineyard objects with the name matching the regex pattern
  vineyardctl ls objects --pattern "vineyard::Tensor<.*>" --regex --ipc-socket /var/run/vineyard.sock

  # List vineyard objects and output as json format
  vineyardctl ls objects --format json --ipc-socket /var/run/vineyard.sock

  # List vineyard objects sorted by the typename
  vineyardctl ls objects --sorted-key typename --limit 1000 --ipc-socket /var/run/vineyard.sock
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for objects
      --ipc-socket string        vineyard IPC socket path
  -l, --limit int                maximum number of objects to return (default 5)
  -p, --pattern string           string that will be matched against the object’s typenames (default "*")
      --port int                 the port of vineyard deployment (default 9600)
  -r, --regex                    regex pattern to match the object’s typenames
      --rpc-socket string        vineyard RPC socket path
  -k, --sorted-key string        key to sort the objects, support:
                                 - id: object id, the default value.
                                 - typename: object typename, e.g. tensor, dataframe, etc.
                                 - type: object type, e.g. global, local, etc.
                                 - instance_id: object instance id. (default "id")
```

## `vineyardctl manager`

Start the manager of vineyard operator

```
vineyardctl manager [flags]
```

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.

### Examples

```shell
  # start the manager of vineyard operator with default configuration
  # (Enable the controller, webhooks and scheduler)
  vineyardctl manager

  # start the manager of vineyard operator without webhooks
  vineyardctl manager --enable-webhook=false

  # start the manager of vineyard operator without scheduler
  vineyardctl manager --enable-scheduler=false

  # only start the controller
  vineyardctl manager --enable-webhook=false --enable-scheduler=false
```

### Options

```
      --enable-scheduler                   Enable scheduler for controller manager. (default true)
      --enable-webhook                     Enable webhook for controller manager. (default true)
      --health-probe-bind-address string   The address the probe endpoint binds to. (default ":8081")
  -h, --help                               help for manager
      --leader-elect                       Enable leader election for controller manager. Enabling this will ensure there is only one active controller manager.
      --metrics-bind-address string        The address the metric endpoint binds to. (default "127.0.0.1:8080")
      --scheduler-config-file string       The location of scheduler plugin's configuration file. (default "/etc/kubernetes/scheduler.yaml")
      --webhook-cert-dir string            The directory to store the generated certificates. (default "/etc/webhook/certs")
```

## `vineyardctl put`

Put basic data type into vineyard

### Synopsis

Put basic data type into vineyard and only support IPC socket.
It receives the flag --value as string and will print object id if succeed.
If you don't specify the ipc socket every time, you can set it as the 
environment variable VINEYARD_IPC_SOCKET.

```
vineyardctl put [flags]
```

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.

### Examples

```shell
  # put value into vineyard with the given ipc socket
  vineyardctl put --value 12345 --ipc-socket /var/run/vineyard.sock
  vineyardctl put --value hello,world --ipc-socket /var/run/vineyard.sock
  
  # If you set the environment variable VINEYARD_IPC_SOCKET
  # you can use the following command to get vineyard blob
  vineyardctl put --value 12345
```

### Options

```
      --deployment-name string   the name of vineyard deployment
  -o, --format string            the output format, support table or json, default is table (default "table")
      --forward-port int         the forward port of vineyard deployment (default 9600)
  -h, --help                     help for put
      --ipc-socket string        vineyard IPC socket path
      --port int                 the port of vineyard deployment (default 9600)
      --rpc-socket string        vineyard RPC socket path
      --value string             vineyard blob value
```

## `vineyardctl schedule`

Schedule a workload or a workflow to existing vineyard cluster.

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl schedule workflow](#vineyardctl-schedule-workflow)	 - Schedule a workflow based on the vineyard cluster
* [vineyardctl schedule workload](#vineyardctl-schedule-workload)	 - Schedule the workload to a vineyard cluster

### Examples

```shell
  # Schedule a workload to a vineyard cluster
  # it will add PodAffinity to the workload
  vineyardctl schedule workload --resource '{kubernetes workload json string}'

  # schedule a workflow to the vineyard cluster
  # it will use the best-effort scheduling strategy
  vineyardctl schedule workflow --file workflow.yaml
```

### Options

```
  -h, --help   help for schedule
```

## `vineyardctl schedule workflow`

Schedule a workflow based on the vineyard cluster

### Synopsis

Schedule a workflow based on the vineyard cluster.
It will apply the workflow to kubernetes cluster and deploy the workload
of the workflow on the vineyard cluster with the best-fit strategy.

```
vineyardctl schedule workflow [flags]
```

**SEE ALSO**

* [vineyardctl schedule](#vineyardctl-schedule)	 - Schedule a workload or a workflow to existing vineyard cluster.

### Examples

```shell
  # schedule a workflow to the vineyard cluster with the best-fit strategy
  vineyardctl schedule workflow --file workflow.yaml
  
  # schedule a workflow without CRD installed
  # Notice, it only works for the workflow built by pods
  vineyardctl schedule workflow --file pod-workflow.yaml --without-crd
```

### Options

```
  -f, --file string   the path of workflow file
  -h, --help          help for workflow
      --without-crd   whether the CRD(especially for GlobalObject and LocalObject) is installed
```

## `vineyardctl schedule workload`

Schedule the workload to a vineyard cluster

### Synopsis

Schedule the workload to a vineyard cluster.
It will add the podAffinity to the workload so that the workload
will be scheduled to the vineyard cluster. Besides, if the workload
does not have the socket volumeMount and volume, it will add one.

Assume you have the following workload yaml:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-client
  # Notice, you must set the namespace here
  namespace: vineyard-job
spec:
  selector:
    matchLabels:
      app: python
  template:
    metadata:
      labels:
        app: python
    spec:
      containers:
      - name: python
        image: python:3.10
        command: ["python", "-c", "import time; time.sleep(100000)"]
```

Then you can run the following command to add the podAffinity and socket volume 
to the workload yaml:

$ vineyard schedule workload -f workload.yaml -o yaml

After that, you will get the following workload yaml: 

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  creationTimestamp: null
  name: python-client
  namespace: vineyard-job
spec:
  selector:
    matchLabels:
      app: python
  strategy: {}
  template:
   metadata:
      creationTimestamp: null
      labels:
        app: python
    spec:
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app.kubernetes.io/instance
                operator: In
                values:
                - vineyard-system-vineyardd-sample
            namespaces:
            - vineyard-system
            topologyKey: kubernetes.io/hostname

      containers:
      - command:
        - python
        - -c
        - import time; time.sleep(100000)
        env:
        - name: VINEYARD_IPC_SOCKET
          value: /var/run/vineyard.sock
        image: python:3.10
        name: python
        resources: {}
        volumeMounts:
        - mountPath: /var/run
          name: vineyard-socket
      volumes:
      - hostPath:
          path: /var/run/vineyard-kubernetes/vineyard-system/vineyardd-sample
        name: vineyard-socket
```

```
vineyardctl schedule workload [flags]
```

**SEE ALSO**

* [vineyardctl schedule](#vineyardctl-schedule)	 - Schedule a workload or a workflow to existing vineyard cluster.

### Examples

```shell
  # Add the podAffinity to the workload yaml
  vineyardctl schedule workload -f workload.yaml \
  --vineyardd-name vineyardd-sample \
  --vineyardd-namespace vineyard-system

  # Add the podAffinity to the workload for the specific vineyard cluster
  vineyardctl schedule workload --resource '{
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
      "name": "web-server"
    },
    "spec": {
      "selector": {
      "matchLabels": {
        "app": "web-store"
      }
      },
      "replicas": 3,
      "template": {
      "metadata": {
        "labels": {
        "app": "web-store"
        }
      },
      "spec": {
        "affinity": {
        "podAntiAffinity": {
          "requiredDuringSchedulingIgnoredDuringExecution": [
          {
            "labelSelector": {
            "matchExpressions": [
              {
              "key": "app",
              "operator": "In",
              "values": [
                "web-store"
              ]
              }
            ]
            },
            "topologyKey": "kubernetes.io/hostname"
          }
          ]
        },
        "podAffinity": {
          "requiredDuringSchedulingIgnoredDuringExecution": [
          {
            "labelSelector": {
            "matchExpressions": [
              {
              "key": "app",
              "operator": "In",
              "values": [
                "store"
              ]
              }
            ]
            },
            "topologyKey": "kubernetes.io/hostname"
          }
          ]
        }
        },
        "containers": [
        {
          "name": "web-app",
          "image": "nginx:1.16-alpine"
        }
        ]
      }
      }
    }
    }' \
    --vineyardd-name vineyardd-sample \
    --vineyardd-namespace vineyard-system
```

### Options

```
  -f, --file string                  the file path of workload
  -h, --help                         help for workload
  -o, --output string                the output format for vineyardctl schedule workload command (default "json")
      --resource string              the json string of kubernetes workload
      --vineyardd-name string        the namespace of vineyard cluster (default "vineyardd-sample")
      --vineyardd-namespace string   the namespace of vineyard cluster (default "vineyard-system")
```
