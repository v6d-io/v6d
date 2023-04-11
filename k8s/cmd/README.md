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
* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes
* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes
* [vineyardctl inject](#vineyardctl-inject)	 - Inject the vineyard sidecar container into a workload
* [vineyardctl manager](#vineyardctl-manager)	 - Start the manager of vineyard operator
* [vineyardctl schedule](#vineyardctl-schedule)	 - Schedule a workload or a workflow to existing vineyard cluster.

## Options

```
      --create-namespace    create the namespace if it does not exist, default false
  -j, --dump-usage          Dump usage in JSON
  -g, --gen-doc             Generate reference docs, e.g., "./cmd/README.md"
  -h, --help                help for vineyardctl
      --kubeconfig string   kubeconfig path for the kubernetes cluster (default "$HOME/.kube/config")
  -n, --namespace string    the namespace for operation (default "vineyard-system")
  -v, --version             version for vineyardctl
      --wait                wait for the kubernetes resource to be ready, default true (default true)
```

## `vineyardctl create`

Create a vineyard jobs on kubernetes

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl create backup](#vineyardctl-create-backup)	 - Backup the current vineyard cluster on kubernetes
* [vineyardctl create operation](#vineyardctl-create-operation)	 - Insert an operation in a workflow based on vineyard cluster
* [vineyardctl create recover](#vineyardctl-create-recover)	 - Recover the current vineyard cluster on kubernetes

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

Backup the current vineyard cluster on kubernetes

### Synopsis

Backup the current vineyard cluster on kubernetes. You could
backup all objects of the current vineyard cluster quickly.
For persistent storage, you could specify the pv spec and pv
spec.

```
vineyardctl create backup [flags]
```

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes

### Examples

```shell
  # create a backup job for the vineyard cluster on kubernetes
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

  # create a backup job for the vineyard cluster on kubernetes
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

  # create a backup job for the vineyard cluster on kubernetes
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
      --limit int                    the limit of objects to backup (default 1000)
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

Recover the current vineyard cluster on kubernetes

### Synopsis

Recover the current vineyard cluster on kubernetes. You could
recover all objects from a backup of vineyard cluster. Usually,
the recover job should be created in the same namespace of
the backup job.

```
vineyardctl create recover [flags]
```

**SEE ALSO**

* [vineyardctl create](#vineyardctl-create)	 - Create a vineyard jobs on kubernetes

### Examples

```shell
  # create a recover job for a backup job in the same namespace
  vineyardctl create recover --backup-name vineyardd-sample -n vineyard-system
```

### Options

```
      --backup-name string    the name of backup job (default "vineyard-backup")
  -h, --help                  help for recover
      --recover-name string   the name of recover job (default "vineyard-recover")
```

## `vineyardctl delete`

Delete the vineyard components from kubernetes

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.
* [vineyardctl delete backup](#vineyardctl-delete-backup)	 - Delete the backup job on kubernetes
* [vineyardctl delete cert-manager](#vineyardctl-delete-cert-manager)	 - Delete the cert-manager on kubernetes
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

  # delete the default cert-manager on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete cert-manager

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

## `vineyardctl delete cert-manager`

Delete the cert-manager on kubernetes

### Synopsis

Delete the cert-manager in the cert-manager namespace. 
The default version of cert-manager is v1.9.1.

```
vineyardctl delete cert-manager [flags]
```

**SEE ALSO**

* [vineyardctl delete](#vineyardctl-delete)	 - Delete the vineyard components from kubernetes

### Examples

```shell
  # delete the default version(v1.9.1) of cert-manager
  vineyardctl --kubeconfig $HOME/.kube/config delete cert-manager
```

### Options

```
  -h, --help   help for cert-manager
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

  # delete the specific version of vineyard operator in the vineyard-system namespace
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete operator -v 0.12.2

  # delete the vineyard operator from local kustomize dir in the vineyard-system namespace
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config delete operator \
    --local ../config/default
```

### Options

```
  -h, --help             help for operator
  -l, --local string     the local kustomize dir
  -v, --version string   the version of kustomize dir from github repo (default "dev")
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
  -h, --help   help for recover
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
* [vineyardctl deploy cert-manager](#vineyardctl-deploy-cert-manager)	 - Deploy the cert-manager on kubernetes
* [vineyardctl deploy operator](#vineyardctl-deploy-operator)	 - Deploy the vineyard operator on kubernetes
* [vineyardctl deploy vineyard-cluster](#vineyardctl-deploy-vineyard-cluster)	 - Deploy the vineyard cluster from kubernetes
* [vineyardctl deploy vineyard-deployment](#vineyardctl-deploy-vineyard-deployment)	 - DeployVineyardDeployment builds and deploy the yaml file of vineyardd without vineyard operator
* [vineyardctl deploy vineyardd](#vineyardctl-deploy-vineyardd)	 - Deploy the vineyardd on kubernetes

### Examples

```shell
  # deploy the default vineyard cluster on kubernetes
  vineyardctl --kubeconfig $HOME/.kube/config deploy vineyard-cluster

  # deploy the vineyard operator on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy operator

  # deploy the cert-manager on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy cert-manager

  # deploy the vineyardd on kubernetes
  vineyardctl -n vineyard-system --kubeconfig $HOME/.kube/config deploy vineyardd
```

### Options

```
  -h, --help   help for deploy
```

## `vineyardctl deploy cert-manager`

Deploy the cert-manager on kubernetes

### Synopsis

Deploy the cert-manager in the cert-manager namespace. The default
version of cert-manager is v1.9.1.

```
vineyardctl deploy cert-manager [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # install the default version(v1.9.1) in the cert-manager namespace
  # wait for the cert-manager to be ready(default option)
  vineyardctl --kubeconfig $HOME/.kube/config deploy cert-manager

  # install the default version(v1.9.1) in the cert-manager namespace
  # not to wait for the cert-manager to be ready, but we does not recommend
  # to do this, because there may be errors caused by the cert-manager
  # not ready
  vineyardctl --kubeconfig $HOME/.kube/config deploy cert-manager \
    --wait=false
```

### Options

```
  -h, --help   help for cert-manager
```

## `vineyardctl deploy operator`

Deploy the vineyard operator on kubernetes

### Synopsis

Deploy the vineyard operator on kubernetes. You could specify a
stable or development version of the operator. The default
kustomize dir is development version from github repo. Also, you
can install the stable version from github repo or a local
kustomize dir. Besides, you can also  deploy the vineyard
operator in an existing namespace.

```
vineyardctl deploy operator [flags]
```

**SEE ALSO**

* [vineyardctl deploy](#vineyardctl-deploy)	 - Deploy the vineyard components on kubernetes

### Examples

```shell
  # install the development version in the vineyard-system namespace
  # the default kustomize dir is the development version from github repo
  # (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false)
  # and the default namespace is vineyard-system
  # wait for the vineyard operator to be ready(default option)
  vineyardctl deploy operator

  # not to wait for the vineyard operator to be ready
  vineyardctl deploy operator --wait=false

  # install the stable version from github repo in the test namespace
  # the kustomize dir is
  # (https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false&ref=v0.12.2)
  vineyardctl -n test --kubeconfig $HOME/.kube/config deploy operator -v 0.12.2

  # install the local kustomize dir
  vineyardctl --kubeconfig $HOME/.kube/config deploy operator --local ../config/default
```

### Options

```
  -h, --help             help for operator
  -l, --local string     the local kustomize dir
  -v, --version string   the version of kustomize dir from github repo (default "dev")
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
  deploy vineyard-deployment --image vineyardd:v0.12.2
```

### Options

```
  -f, --file string                               the path of vineyardd
  -h, --help                                      help for vineyard-deployment
  -l, --label string                              label of the vineyardd
      --name string                               the name of vineyardd (default "vineyardd-sample")
      --plugin.backupImage string                 the backup image of vineyardd (default "ghcr.io/v6d-io/v6d/backup-job")
      --plugin.daskRepartitionImage string        the dask repartition image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/dask-repartition")
      --plugin.distributedAssemblyImage string    the distributed image of vineyard workflow (default "ghcr.io/v6d-io/v6d/distributed-assembly")
      --plugin.localAssemblyImage string          the local assembly image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/local-assembly")
      --plugin.recoverImage string                the recover image of vineyardd (default "ghcr.io/v6d-io/v6d/recover-job")
      --vineyard.etcd.replicas int                the number of etcd replicas in a vineyard cluster (default 3)
      --vineyard.replicas int                     the number of vineyardd replicas (default 3)
      --vineyardd.envs strings                    The environment variables of vineyardd
      --vineyardd.etcdEndpoint string             The etcd endpoint of vineyardd (default "http://etcd-for-vineyard:2379")
      --vineyardd.etcdPrefix string               The etcd prefix of vineyardd (default "/vineyard")
      --vineyardd.image string                    the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --vineyardd.imagePullPolicy string          the imagePullPolicy of vineyardd (default "IfNotPresent")
      --vineyardd.metric.enable                   enable metrics of vineyardd
      --vineyardd.metric.image string             the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --vineyardd.metric.imagePullPolicy string   the imagePullPolicy of the metric image (default "IfNotPresent")
      --vineyardd.service.port int                the service port of vineyard service (default 9600)
      --vineyardd.service.selector string         the service selector of vineyard service (default "rpc.vineyardd.v6d.io/rpc=vineyard-rpc")
      --vineyardd.service.type string             the service type of vineyard service (default "ClusterIP")
      --vineyardd.size string                     The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.  (default "256Mi")
      --vineyardd.socket string                   The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --vineyardd.spill.config string             If you want to enable the spill mechanism, please set the name of spill config
      --vineyardd.spill.path string               The path of spill config
      --vineyardd.spill.pv-pvc-spec string        the json string of the persistent volume and persistent volume claim
      --vineyardd.spill.spillLowerRate string     The low watermark of spilling memory (default "0.3")
      --vineyardd.spill.spillUpperRate string     The high watermark of spilling memory (default "0.8")
      --vineyardd.streamThreshold int             memory threshold of streams (percentage of total memory) (default 80)
      --vineyardd.syncCRDs                        enable metrics of vineyardd (default true)
      --vineyardd.volume.mountPath string         Set the mount path for the pvc
      --vineyardd.volume.pvcname string           Set the pvc name for storing the vineyard objects persistently, 
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
  -f, --file string                               the path of vineyardd
  -h, --help                                      help for vineyardd
      --name string                               the name of vineyardd (default "vineyardd-sample")
      --plugin.backupImage string                 the backup image of vineyardd (default "ghcr.io/v6d-io/v6d/backup-job")
      --plugin.daskRepartitionImage string        the dask repartition image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/dask-repartition")
      --plugin.distributedAssemblyImage string    the distributed image of vineyard workflow (default "ghcr.io/v6d-io/v6d/distributed-assembly")
      --plugin.localAssemblyImage string          the local assembly image of vineyardd workflow (default "ghcr.io/v6d-io/v6d/local-assembly")
      --plugin.recoverImage string                the recover image of vineyardd (default "ghcr.io/v6d-io/v6d/recover-job")
      --vineyard.etcd.replicas int                the number of etcd replicas in a vineyard cluster (default 3)
      --vineyard.replicas int                     the number of vineyardd replicas (default 3)
      --vineyardd.envs strings                    The environment variables of vineyardd
      --vineyardd.etcdEndpoint string             The etcd endpoint of vineyardd (default "http://etcd-for-vineyard:2379")
      --vineyardd.etcdPrefix string               The etcd prefix of vineyardd (default "/vineyard")
      --vineyardd.image string                    the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --vineyardd.imagePullPolicy string          the imagePullPolicy of vineyardd (default "IfNotPresent")
      --vineyardd.metric.enable                   enable metrics of vineyardd
      --vineyardd.metric.image string             the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --vineyardd.metric.imagePullPolicy string   the imagePullPolicy of the metric image (default "IfNotPresent")
      --vineyardd.service.port int                the service port of vineyard service (default 9600)
      --vineyardd.service.selector string         the service selector of vineyard service (default "rpc.vineyardd.v6d.io/rpc=vineyard-rpc")
      --vineyardd.service.type string             the service type of vineyard service (default "ClusterIP")
      --vineyardd.size string                     The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.  (default "256Mi")
      --vineyardd.socket string                   The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --vineyardd.spill.config string             If you want to enable the spill mechanism, please set the name of spill config
      --vineyardd.spill.path string               The path of spill config
      --vineyardd.spill.pv-pvc-spec string        the json string of the persistent volume and persistent volume claim
      --vineyardd.spill.spillLowerRate string     The low watermark of spilling memory (default "0.3")
      --vineyardd.spill.spillUpperRate string     The high watermark of spilling memory (default "0.8")
      --vineyardd.streamThreshold int             memory threshold of streams (percentage of total memory) (default 80)
      --vineyardd.syncCRDs                        enable metrics of vineyardd (default true)
      --vineyardd.volume.mountPath string         Set the mount path for the pvc
      --vineyardd.volume.pvcname string           Set the pvc name for storing the vineyard objects persistently, 
```

## `vineyardctl inject`

Inject the vineyard sidecar container into a workload

### Synopsis

Inject the vineyard sidecar container into a workload. You can
get the injected workload yaml and some etcd yaml from the output.

```
vineyardctl inject [flags]
```

**SEE ALSO**

* [vineyardctl](#vineyardctl)	 - vineyardctl is the command-line tool for interact with the Vineyard Operator.

### Examples

```shell
  # inject the default vineyard sidecar container into a workload
  vineyardctl inject -f workload.yaml | kubectl apply -f -
```

### Options

```
      --etcd-replicas int                       the number of etcd replicas (default 1)
  -f, --file string                             The yaml of workload
  -h, --help                                    help for inject
      --sidecar.envs strings                    The environment variables of vineyardd
      --sidecar.etcdEndpoint string             The etcd endpoint of vineyardd (default "http://etcd-for-vineyard:2379")
      --sidecar.etcdPrefix string               The etcd prefix of vineyardd (default "/vineyard")
      --sidecar.image string                    the image of vineyardd (default "vineyardcloudnative/vineyardd:latest")
      --sidecar.imagePullPolicy string          the imagePullPolicy of vineyardd (default "IfNotPresent")
      --sidecar.metric.enable                   enable metrics of vineyardd
      --sidecar.metric.image string             the metic image of vineyardd (default "vineyardcloudnative/vineyard-grok-exporter:latest")
      --sidecar.metric.imagePullPolicy string   the imagePullPolicy of the metric image (default "IfNotPresent")
      --sidecar.service.port int                the service port of vineyard service (default 9600)
      --sidecar.service.selector string         the service selector of vineyard service (default "rpc.vineyardd.v6d.io/rpc=vineyard-rpc")
      --sidecar.service.type string             the service type of vineyard service (default "ClusterIP")
      --sidecar.size string                     The size of vineyardd. You can use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.  (default "256Mi")
      --sidecar.socket string                   The directory on host for the IPC socket file. The namespace and name will be replaced with your vineyard config (default "/var/run/vineyard-kubernetes/{{.Namespace}}/{{.Name}}")
      --sidecar.spill.config string             If you want to enable the spill mechanism, please set the name of spill config
      --sidecar.spill.path string               The path of spill config
      --sidecar.spill.pv-pvc-spec string        the json string of the persistent volume and persistent volume claim
      --sidecar.spill.spillLowerRate string     The low watermark of spilling memory (default "0.3")
      --sidecar.spill.spillUpperRate string     The high watermark of spilling memory (default "0.8")
      --sidecar.streamThreshold int             memory threshold of streams (percentage of total memory) (default 80)
      --sidecar.syncCRDs                        enable metrics of vineyardd (default true)
      --sidecar.volume.mountPath string         Set the mount path for the pvc
      --sidecar.volume.pvcname string           Set the pvc name for storing the vineyard objects persistently, 
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
```

### Options

```
  -f, --file string   the path of workflow file
  -h, --help          help for workflow
```

## `vineyardctl schedule workload`

Schedule the workload to a vineyard cluster

### Synopsis

Schedule the workload to a vineyard cluster.
It will add the podAffinity to the workload so that the workload
will be scheduled to the vineyard cluster.

```
vineyardctl schedule workload [flags]
```

**SEE ALSO**

* [vineyardctl schedule](#vineyardctl-schedule)	 - Schedule a workload or a workflow to existing vineyard cluster.

### Examples

```shell
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
  -h, --help                         help for workload
      --resource string              the json string of kubernetes workload
      --vineyardd-name string        the namespace of vineyard cluster (default "vineyardd-sample")
      --vineyardd-namespace string   the namespace of vineyard cluster (default "vineyard-system")
```

