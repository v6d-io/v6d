# vineyard operator charts

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard-operator)

A helm chart for [vineyard operator][3], which manages all relavant components about vineyard.

## Install

Vineyard operator has been integrated with [Helm](https://helm.sh/). Add the vineyard repository to your Helm client:

```bash

$ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
$ helm repo update

```

Refer to the [helm repo](https://helm.sh/docs/helm/helm_repo/) for more repo information.

The webhook is enabled by default, please make sure you have the [Cert-Manager](https://cert-manager.io/docs/installation/) installed, then deploy it in the `default` namespace as follows:

```bash

$ helm install vineyard-operator vineyard/vineyard-operator

```

If you want to deploy it in a specific namespace, you can use the `--namespace` option:

```bash

$ helm install vineyard-operator vineyard/vineyard-operator \
      --namespace vineyard-system
   
```

If you want to set the value of the chart, you can use the `--set` option:

```bash

$ helm install vineyard-operator vineyard/vineyard-operator \
      --set image.tag=v0.10.1

```

Refer to the [helm install](https://helm.sh/docs/helm/helm_install/) for more command information.

Install `vineyardd` as follows.

```bash

$ cat <<EOF | kubectl apply -f -
apiVersion: k8s.v6d.io/v1alpha1
kind: Vineyardd
metadata:
  name: vineyardd-sample
  # don't use default namespace
  namespace: vineyard-system
spec:
  image: ghcr.io/v6d-io/v6d/vineyardd:alpine-latest
  replicas: 2
  imagePullPolicy: IfNotPresent
  syncCRDs: true
  enableMetrics: false
  etcd:
    replicas: 3
  service:
    type: ClusterIP
    port: 9600
EOF
   
```

## Uninstall

The installed charts can be removed with

```shell

$ helm uninstall vineyard-operator

More information about the helm chart could be found at [artifacthub][1] and [parameters][2].

## Values

The following table lists the configurable parameters of the Vineyard Operator chart and their default values. Besides, you can refer the [doc](https://v6d.io/notes/vineyard-operator.html) to get more detail about the vineyard operator.

| Key                | Type   | Default                                                                   | Description                                                                                                                |
| ------------------ | ------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| replicaCount       | int    | 1                                                                         | The replica of vineyard operator.                                                                                          |
| image.repository   | string | "vineyardcloudnative/vineyard-operator"                                   | The repository of vineyard operator image.                                                                                 |
| image.pullPolicy   | string | "IfNotPresent"                                                            | The pull policy of vineyard operator image.                                                                                |
| image.tag          | string | "latest"                                                                  | The image tag of vineyard operator image.                                                                                  |
| webhook.enabled | bool | true                                             | Enable the webhook. If you only want to deploy the vineyard, set false here.                                                                              |
| webhook.port | int | 9443                                             | The port of the webhook in vineyard operator.                                                 |
| serviceAccountName | string | "Vineyard-manager"                                                        | The service account name of vineyard operator.                                                                             |
| service.type       | string | "ClusterIP"                                                               | The type of the service.                                                                                                   |
| service.port       | int    | 9600                                                                      | The internal port of vineyard operator service                                                                             |
| resources          | object | {limits: {cpu: 500m, memory:500Mi}},{requests: {cpu: 500m, memory:500Mi}} | The limits and requests of vineyard operator.                                                                              |
| tolerations        | object | {}                                                                        | Tolerations allow the scheduler to schedule pods with matching taints                                                      |
| affinity           | object | {}                                                                        | Affinity enables the scheduler to place a pod either on a group of nodes or a pod relative to the placement of other pods. |

## License

**vineyard** is distributed under [Apache License 2.0](https://github.com/v6d-io/v6d/blob/main/LICENSE).
Please note that third-party libraries may not have the same license as vineyard.

[1]: https://artifacthub.io/packages/helm/vineyard/vineyard-operator

[2]: https://github.com/v6d-io/v6d/blob/main/charts/vineyard-operator/values.yaml

[3]: https://github.com/v6d-io/v6d/k8s
