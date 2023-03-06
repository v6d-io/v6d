## vineyardctl

`vineyardctl` is the command-line tool for working with the Vineyard Operator. It supports creating, deleting and checking status of Vineyard Operator. It also supports managing the vineyard relevant components such as vineyardd and pluggable drivers. Each function is implemented as a subcommand of `vineyardctl`.

You could use the following command to build the `vineyardctl`, please make sure you have a golang environment already.
```
$ go build -o vineyardctl k8s/cmd/main.go
$ ./vineyardctl --help
```

### GlobalFlags

The following flags are available for the sub commands:

`--kubeconfig`: the flag is used to set the path for the kubernetes cluster configuration file. The default path is
`$HOME/.kube/config`.

`--namespace`: the flag is used to set the namespace for the current kubernetes object. The default value is `vineyard-system`, make sure
you have already created it.

### SubCommands

#### deploy

`deploy` is a subcommand of `vineyardctl` which is used to deploy different vineyard components on kubernetes, the following components can be deployed by the subcommand.

- `cert-manager`. Before installing the operator, you should install the cert-manager first. The cert-manager is used to generate
the TLS certificates for the next operator.

- `operator`. The operator is used to manage the vineyard cluster and
vineyard jobs on kubernetes. It contains three components: vineyard controllers, vineyard webhook, and vineyard scheduler. Also, 
while deploying the operator, the CRDs will be created automatically.

- `vineyardd`. The vineyardd is a CRD object, it is used to create a vineyardd instance on kubernetes. After the vineyardd is created,
several kubernetes resources will be created automatically, such as the etcd pod and service, the vineyard deployment, and related rpc service.

- `vineyard-cluster`. The vineyard-cluster is a summary of the above components, it is used to create a vineyard cluster on kubernetes.
During the creation, the cert-manager, operator and vineyardd will be deployed step by step.

- `vineyard-deployment`. While you don't want to install an operator, you can use the vineyard-deployment to deploy the vineyard cluster directly.
It will create the same kubernetes resources as the vineyardd, but it doesn't need to install the operator. However, the vineyard-deployment can only be managed by the users themselves.

For more information about the vineyard components, please refer to the [vineyard operator doc](https://github.com/v6d-io/v6d/blob/main/docs/notes/vineyard-operator.rst#vineyard-operator).

#### create

`create` is a subcommand of `vineyardctl` for creating a vineyard job on kubernetes. It can be used to create a backup job, an operation
job and a recover job. 

- `backup`. A backup job can be used to back up the current vineyard cluster on Kubernetes.
- `operation`. The operation job can be used to insert an operation in a workflow based on the vineyard cluster.
- `recover`. The recover job can be used to recover the vineyard cluster from a specific backup state.

For more information about the job, please refer to the [operation and drivers](https://github.com/v6d-io/v6d/blob/main/docs/notes/vineyard-operator.rst#operations-and-drivers) and the [failover mechanism](https://github.com/v6d-io/v6d/blob/main/docs/notes/vineyard-operator.rst#failover-mechanism-of-vineyard-cluster).

#### delete

`delete` is used to delete the vineyard components and vineyard jobs on kubernetes, the following components can be deleted by the subcommand.

Vineyard components:

- cert-manager
- operator
- vineyardd
- vineyard-cluster
- vineyard-deployment

Vineyard jobs:

- backup
- operation
- recover

#### inject

`inject` is a subcommand of `vineyardctl` that allows users to inject a Vineyard sidecar into their YAML file. This is useful for workloads that are based on Vineyard but do not need the scheduler mechanism, as it allows them to share the Vineyard socket provided by the sidecar while running in the same pod. Using this subcommand, users can inject the Vineyard sidecar into their YAML file and output a new YAML file, which can then be used to create the workload with Vineyard on Kubernetes.

#### manager

`manager` is a subcommand of `vineyardctl` that provides users with the ability to configure and start components of the Vineyard operator, such as vineyard-controllers, vineyard-webhook, and vineyard-scheduler. This enables users to conveniently manage and launch components of the operator, allowing them to customize their deployment according to their needs.


#### schedule

`schedule` is a subcommand of `vineyardctl` that schedules workloads to the Vineyard cluster. By providing a workload JSON string and a Vineyard cluster, this subcommand adds a PodAffinity to the workload, allowing it to be automatically scheduled to the cluster. This PodAffinity helps to ensure workloads are distributed evenly across the cluster, increasing scalability and availability.

#### References

For more detailed usage, please refer to the [vineyardctl references](references.md).