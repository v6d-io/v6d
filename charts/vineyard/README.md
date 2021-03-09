vineyard charts
===============

[![Artifact HUB](https://img.shields.io/endpoint?url=https://artifacthub.io/badge/repository/vineyard)](https://artifacthub.io/packages/helm/vineyard/vineyard)

Vineyard is an in-memory immutable data manager that provides **out-of-box high-level**
abstraction and **zero-copy in-memory** sharing for distributed data in big data tasks,
such as numerical computing, machine learning, and graph analytics.

Install
-------

Vineyard is an has been integrated with [Helm](https://helm.sh/). Deploy vineyard as
a `DaemonSet` using `helm`:

```bash
helm repo add vineyard https://dl.bintray.com/libvineyard/charts/
helm install vineyard vineyard/vineyard
```

More information about the helm chart could be found at [artifacthub](https://artifacthub.io/packages/helm/vineyard/vineyard).

License
-------

**libvineyard** is distributed under [Apache License 2.0](https://github.com/alibaba/libvineyard/blob/main/LICENSE).
Please note that third-party libraries may not have the same license as libvineyard.
