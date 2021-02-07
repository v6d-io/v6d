Vineyard Operator
-----------------

Vineyard operator is a kubernetes controller to operator the vineyard cluster. Within the
vineyard operator, there's a scheduler-plugin called `vineyard-scheduler` which performs
data aware job scheduling based on the locality information of objects in vineyard.

### CRDs

Vineyard operator defines two CRDs to represents objects in vineyard:

- `GlobalObject` for global objects
- `LocalObject` for local objects

### Scheduler Plugin

The scheduler plugin works based the CRDs in the cluster. When vineyardd persists objects
to etcd, it will create a CRD entry in kubernetes as well. The metadata of CRD has location
information and scheduler plugin scores pod based on the data locality of required objects
by jobs. For jobs, some label annotation are used to describe which objects are required
in the pod spec.

```yaml
labels:
    app: pytorch
    scheduling.k8s.v6d.io/job: pytorch
    scheduling.k8s.v6d.io/required: "o0000022285553e22"
    scheduling.k8s.v6d.io/replica: "4"
```

### Deploy

To make the docker image for the controller, run

```bash
make docker-build
```

To install CRDs and required k8s cluster roles, run

```bash
make predeploy
```

To deploy the custom scheduler, run

```bash
make deploy
```
