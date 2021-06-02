vineyard charts
===============

![Vineyard](https://v6d.io/_static/vineyard-logo.png)

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard)

Vineyard is an in-memory immutable data manager that provides **out-of-box high-level**
abstraction and **zero-copy in-memory** sharing for distributed data in big data tasks,
such as numerical computing, machine learning, and graph analytics.

Install
-------

Vineyard is an has been integrated with [Helm](https://helm.sh/). Deploy vineyard as
a `DaemonSet` using `helm`:

```console
helm repo add vineyard https://vineyard.oss-ap-southeast-1.aliyuncs.com/charts/
helm install vineyard vineyard/vineyard
```

Uninstall
---------

The installed charts can be removed with

```console
helm uninstall vineyard
```

More information about the helm chart could be found at [artifacthub][1] and [parameters][2].

License
-------

**vineyard** is distributed under [Apache License 2.0](https://github.com/v6d-io/v6d/blob/main/LICENSE).
Please note that third-party libraries may not have the same license as vineyard.

[1]: https://artifacthub.io/packages/helm/vineyard/vineyard
[2]: https://github.com/v6d-io/v6d/blob/main/charts/vineyard/values.yaml
