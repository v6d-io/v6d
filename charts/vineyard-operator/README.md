# vineyard operator charts

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard-operator)

A helm chart for [vineyard operator][3], which manages all relavant components about vineyard.

## Install

Vineyard operator has been integrated with [Helm](https://helm.sh/). Deploy it as follows:

```console
helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
helm install vineyard-operator vineyard/vineyard-operator
```

Install vineyardd as follows.

.. code:: shell

   curl https://raw.githubusercontent.com/v6d-io/v6d/main/k8s/test/e2e/vineyardd.yaml | kubectl apply -f -

## Uninstall

The installed charts can be removed with

```console
helm uninstall vineyard-operator
```

More information about the helm chart could be found at [artifacthub][1] and [parameters][2].

## Values

The following table lists the configurable parameters of the Vineyard Operator chart and their default values.

| Key                | Type   | Default                                                                   | Description                                                                                                                |
| ------------------ | ------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| replicaCount       | int    | 1                                                                         | The replica of vineyard operator.                                                                                          |
| image.repository   | string | "vineyardcloudnative/vineyard-operator"                                   | The repository of vineyard operator image.                                                                                 |
| image.pullPolicy   | string | "IfNotPresent"                                                            | The pull policy of vineyard operator image.                                                                                |
| image.tag          | string | "latest"                                                                  | The image tag of vineyard operator image.                                                                                  |
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
