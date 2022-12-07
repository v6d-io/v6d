Vineyard Operator
-----------------

To manage all vineyard relevant components in kubernetes cluster, we proposal the vineyard 
operator. With it, users can easily deploy vineyard components, and manage their lifecycle. 
For more details, please refer to [vineyard operator doc](https://v6d.io/notes/vineyard-operator.html).

### CRDs

Vineyard operator defines the following CRDs to manage vineyard components and operations.

- `GlobalObject` for global objects.
- `LocalObject` for local objects.
- `Vineyardd` for deployment of vineyard.
- `Operation` for inserting operation to a workflow.
- `Sidecar` for vineyard as a sidecar container.

### Deploy

To make the docker image for the controller, run

```bash
make docker-build
```

To install CRDs and required k8s cluster roles, run

```bash
make predeploy
```

To deploy the vineyard operator, run

```bash
make deploy
```
