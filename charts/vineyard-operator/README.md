# vineyard operator charts

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard-operator)

A helm chart for [vineyard operator][3], which manages all relevant components about vineyard.

## Install

Vineyard operator has been integrated with [Helm](https://helm.sh/). Add the vineyard repository to your Helm client:

```shell
$ helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
$ helm repo update
```

Refer to the [helm repo](https://helm.sh/docs/helm/helm_repo/) for more repo information.

Install `vineyard-operator` as follows.

> **NOTE:** DON'T add the flag `--wait` during the helm install, the operator will not be installed successfully if you add it. For more detail, please refer to [issue](https://github.com/v6d-io/v6d/issues/1490).

```shell
$ helm install vineyard-operator vineyard/vineyard-operator
```

If you want to deploy it in a specific namespace, you can use the `--namespace` option:

```shell
$ helm install vineyard-operator vineyard/vineyard-operator \
      --namespace vineyard-system \
      --create-namespace
```

If you want to set the value of the chart, you can use the `--set` option:

```shell
$ helm install vineyard-operator vineyard/vineyard-operator \
      --namespace vineyard-system \
      --set controllerManager.manager.image.tag=v0.10.1
```

Refer to the [helm install](https://helm.sh/docs/helm/helm_install/) for more command information.

Install `vineyardd` as follows.

```shell

$ cat <<EOF | kubectl apply -f -
apiVersion: k8s.v6d.io/v1alpha1
kind: Vineyardd
metadata:
  name: vineyardd-sample
spec:
  replicas: 3
  vineyard:
    image: vineyardcloudnative/vineyardd:latest
    imagePullPolicy: IfNotPresent
EOF
```

## Uninstall

The installed charts can be removed with

```shell

$ helm uninstall vineyard-operator

```
More information about the helm chart could be found at [artifacthub][1] and [parameters][2].

## Values

The following table lists the configurable parameters of the Vineyard Operator chart and their default values. 
Besides, you can refer the [doc](https://v6d.io/notes/cloud-native/vineyard-operator.html) to get more detail about the vineyard operator.

| Key                                              | Type   | Default                                                                   | Description                                  |
|--------------------------------------------------|--------|---------------------------------------------------------------------------|----------------------------------------------|
| controllerManager.kubeRbacProxy.image.repository | string | "gcr.io/kubebuilder/kube-rbac-proxy"                                      | The repository of kubeRbacProxy image.       |
| controllerManager.kubeRbacProxy.image.tag        | string | "v0.13.0"                                                                 | The tag of kubeRbacProxy image.              |
| controllerManager.kubeRbacProxy.resources        | object | {limits: {cpu: 300m, memory:300Mi}},{requests: {cpu: 300m, memory:300Mi}} | The limits and requests of kubeRbacProxy.    |
| controllerManager.manager.image.repository       | string | "gcr.io/kubebuilder/kube-rbac-proxy"                                      | The repository of operator-manager image.    |
| controllerManager.manager.image.tag              | string | "latest"                                                                  | The tag of operator-manager image.           |
| controllerManager.manager.resources              | object | {limits: {cpu: 500m, memory:500Mi}},{requests: {cpu: 500m, memory:500Mi}} | The limits and requests of operator-manager. |
| controllerManager.replicas                       | int    | 1                                                                         | The replica of vineyard operator.            |
| kubernetesClusterDomain                          | string | "cluster.local"                                                           | The domain name of you kubernetes cluster.   |
| metricsService.ports.name                        | string | "https"                                                                   | The name of metrics service.                 |
| metricsService.ports.port                        | int    | "8443"                                                                    | The port of metrics service.                 |
| metricsService.ports.targetPort                  | string | "https"                                                                   | The target port of metrics service.          |
| metricsService.type                              | string | "ClusterIP"                                                               | The type of metrics service.                 |
| webhookService.ports.port                        | int    | "443"                                                                     | The port of webhook service.                 |
| webhookService.ports.protocol                    | string | "TCP"                                                                     | The protocol of webhook service.             |
| webhookService.ports.targetPort                  | string | "9443"                                                                    | The target port of webhook service.          |
| webhookService.type                              | string | "ClusterIP"                                                               | The type of webhook service.                 |

## License

**vineyard** is distributed under [Apache License 2.0](https://github.com/v6d-io/v6d/blob/main/LICENSE).
Please note that third-party libraries may not have the same license as vineyard.

[1]: https://artifacthub.io/packages/helm/vineyard/vineyard-operator

[2]: https://github.com/v6d-io/v6d/blob/main/charts/vineyard-operator/values.yaml

[3]: https://github.com/v6d-io/v6d/k8s
