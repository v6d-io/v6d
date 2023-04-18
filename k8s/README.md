Vineyard Operator
-----------------

To manage all vineyard relevant components in kubernetes cluster, we proposal the vineyard 
operator. With it, users can easily deploy vineyard components, and manage their lifecycle. 
For more details, please refer to [vineyard operator doc](https://v6d.io/notes/cloud-native/vineyard-operator.html). 
At present, the vineyard operator is at the beginning stage, and the main milestones are as follows.

### Short milestones

- [x] Deployment, management, and observability of vineyard components.
- [ ] Integrate with several Kubernetes workflow engines such as Argo and Kubeflow.

### Long miletones

- [ ] Multi-tenant management and authentication management.
- [ ] Supports deployment on the production environment(Aliyun, AWS, Azure, etc.).



### CRDs

Vineyard operator defines the following CRDs to manage vineyard components and operations.

*   `GlobalObject` for global objects.
*   `LocalObject` for local objects.
*   `Vineyardd` for deployment of vineyard.
*   `Operation` for inserting operation to a workflow.
*   `Sidecar` for vineyard as a sidecar container.

### Deploy with remote directory

You could use the kustomize to build the remote directories on the github and apply the manifests
as follows.

```bash
$ kustomize build https://github.com/v6d-io/v6d/k8s/config/default\?submodules=false | kubectl apply -f -
```

### Deploy locally

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

### Test

You could create a default kind cluster and run a specific e2e test locally as follows.

```bash
make -C k8s e2e-tests-assembly-local
```
