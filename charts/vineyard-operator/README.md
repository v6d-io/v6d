# vineyard operator charts

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard-operator)

A helm chart for [vineyard operator][3], which manages all relavant compenents about vineyard.

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

## License

**vineyard** is distributed under [Apache License 2.0](https://github.com/v6d-io/v6d/blob/main/LICENSE).
Please note that third-party libraries may not have the same license as vineyard.

[1]: https://artifacthub.io/packages/helm/vineyard/vineyard-operator

[2]: https://github.com/v6d-io/v6d/blob/main/charts/vineyard-operator/values.yaml

[3]: https://github.com/v6d-io/v6d/k8s
