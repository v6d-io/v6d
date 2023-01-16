# Copyright 2020-2023 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Current Operator version
VERSION ?= 0.0.1
# Default bundle image tag
BUNDLE_IMG ?= controller-bundle:$(VERSION)
# Options for 'bundle-build'
ifneq ($(origin CHANNELS), undefined)
BUNDLE_CHANNELS := --channels=$(CHANNELS)
endif
ifneq ($(origin DEFAULT_CHANNEL), undefined)
BUNDLE_DEFAULT_CHANNEL := --default-channel=$(DEFAULT_CHANNEL)
endif
BUNDLE_METADATA_OPTS ?= $(BUNDLE_CHANNELS) $(BUNDLE_DEFAULT_CHANNEL)

# Image URL to use all building/pushing image targets
IMG ?= vineyardcloudnative/vineyard-operator:latest

temp=$(shell mktemp -d)
# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

GOLINT := $(shell command -v ${GOBIN}/golangci-lint --version || shell ${GOBIN}/golangci-lint --version)
HELMIFY := $(shell command -v ${GOBIN}/helmify --version || shell ${GOBIN}/helmify --version)
E2E := $(shell command -v ${GOBIN}/e2e --version || shell ${GOBIN}/e2e --version)

## Tool Versions
KUSTOMIZE_VERSION ?= v4@v4.5.5
CONTROLLER_TOOLS_VERSION ?= v0.9.2
GOLANGCI_LINT_VERSION ?= v1.49.0
HELMIFY_VERSION ?= v0.3.18
E2E_VERSION ?= 2631e76926604c4e30ca170bed916804c86980b6

all: manager

#check ci locally
check: lint e2e-test
.PHONY: check

# Build a 4-nodes(1 master and 3 workers) kind cluster with local registry
build-local-cluster:
	@touch /tmp/e2e-k8s.config
	@echo "the kubeconfig path is /tmp/e2e-k8s.config"
	@echo "Creating the kind cluster with local registry"
	@bash test/hack/build-kind-cluster-with-local-registry.sh
.PHONY: build-local-cluster

delete-local-cluster:
	@rm /tmp/e2e-k8s.config
	@echo "Deleting the kind cluster with local registry"
	@bash test/hack/delete-kind-cluster-with-local-registry.sh
.PHONY: delete-local-cluster

prepare-e2e-test: e2e
	@echo "Install e2e test dependencies..."
	@sudo bash test/hack/prepare-e2e.sh
	@echo "Build vineyard operator image..."
	@make docker-build
.PHONY: prepare-e2e-test

# install the cert-manager and wait for ready
install-cert-manager:
	@echo "Installing cert-manager..."
	@kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml --kubeconfig=/tmp/e2e-k8s.config
	@kubectl wait pods -n cert-manager --all --for condition=Ready --timeout=300s --kubeconfig=/tmp/e2e-k8s.config
	@echo "Cert-Manager ready."

# install the vineyard-operator and wait for ready
install-vineyard-operator:
	@echo "Installing vineyard-operator..."
	@docker tag vineyardcloudnative/vineyard-operator:latest localhost:5001/vineyard-operator:latest
	@docker push localhost:5001/vineyard-operator:latest
	@export IMG=localhost:5001/vineyard-operator:latest KUBECONFIG=/tmp/e2e-k8s.config && make deploy
	@kubectl wait deployment/vineyard-controller-manager --for condition=Available -n vineyard-system --timeout=300s --kubeconfig=/tmp/e2e-k8s.config
	@echo "Vineyard-Operator Ready"

# install the vineyard cluster and wait for ready
install-vineyard-cluster:
	@echo "Installing vineyard cluster..."
	@kubectl apply -f test/e2e/vineyardd.yaml --kubeconfig=/tmp/e2e-k8s.config
	@kubectl wait vineyardd/vineyardd-sample --for condition=Available -n vineyard-system --timeout=300s --kubeconfig=/tmp/e2e-k8s.config
	@echo "Vineyard cluster Ready"

e2e-tests-assembly-local: kustomize prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running local assembly e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/assembly/local-assembly-e2e.yaml
	@echo "local assembly e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-assembly-local

e2e-tests-assembly-distributed: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running distributed assembly e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/assembly/distributed-assembly-e2e.yaml
	@echo "distributed assembly e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-assembly-distributed

e2e-tests-autogenerated-helm-chart: prepare-e2e-test build-local-cluster install-cert-manager
	@echo "Running autogenerated helm chart e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/autogenerated-helm-chart/e2e.yaml
	@echo "autogenerated helm chart e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-autogenerated-helm-chart

e2e-tests-failover: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running failover e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/failover/e2e.yaml
	@echo "failover e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-failover

e2e-tests-repartition-dask: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running repartition dask e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/repartition/dask-repartition-e2e.yaml
	@echo "repartition dask e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-repartition-dask

e2e-tests-scheduler-outside-cluster: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running scheduler outside cluster e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/scheduling-outside-cluster/e2e.yaml
	@echo "scheduler outside cluster e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-scheduler-outside-cluster

e2e-tests-serialize: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running serialize e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/serialize/e2e.yaml
	@echo "serialize e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-serialize

e2e-tests-sidecar: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator
	@echo "Running sidecar e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/sidecar/e2e.yaml
	@echo "sidecar e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-sidecar

e2e-tests-spill: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator
	@echo "Running spill e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/spill/e2e.yaml
	@echo "spill e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-spill

e2e-tests-workflow: prepare-e2e-test build-local-cluster install-cert-manager install-vineyard-operator install-vineyard-cluster
	@echo "Running workflow e2e test..."
	@cd .. && ${GOBIN}/e2e run --config=k8s/test/e2e/workflow/e2e.yaml
	@echo "workflow e2e test passed."
	@# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	@make delete-local-cluster
.PHONY: e2e-tests-workflow

# Build manager binary
manager: generate fmt
	go build -o bin/manager main.go

run: generate fmt
	go run ./main.go

# Install CRDs into a cluster
install: manifests kustomize
	$(KUSTOMIZE) build config/crd | kubectl apply -f -

# Uninstall CRDs from a cluster
uninstall: manifests kustomize
	$(KUSTOMIZE) build config/crd | kubectl delete -f -

# Deploy controller in the configured Kubernetes cluster in ~/.kube/config
deploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default | kubectl apply -f -

# Undeploy controller in the configured Kubernetes cluster in ~/.kube/config
undeploy: kustomize
	kubectl -n vineyard-system delete deployment vineyard-controller-manager

# Undeploy controller and all created resources in the configured Kubernetes cluster in ~/.kube/config
undeploy-all: kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default | kubectl delete -f -

# Deploy namespace, CRDs, rbac, etc. except the controller in the configured Kubernetes cluster in ~/.kube/config
predeploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}

unpredeploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}

# dump the deployment configuration
dry-run: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=${IMG}
	$(KUSTOMIZE) build config/default > controller.yaml

# Generate manifests e.g. CRD, RBAC etc.
manifests: controller-gen
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd:maxDescLen=0 webhook paths="./..." output:crd:artifacts:config=config/crd/bases

sample: predeploy
	$(KUSTOMIZE) build config/samples | kubectl apply -f -

unsample:
	$(KUSTOMIZE) build config/samples | kubectl delete -f -

# Go modules
modules	= ./apis/... \
		  ./controllers/... \
		  ./pkg/...

# Run golangci-lint
golint: 
    ifeq (${GOLINT},)
		@echo "golangci-lint not exist, installing it..."
		go install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
    endif
	${GOBIN}/golangci-lint run ./...
.PHONY: golint

# Install helmify if not exist
helmify:
    ifeq (${HELMIFY},)
		@echo "helmify not exist, installing it..."
		go install github.com/arttor/helmify/cmd/helmify@$(HELMIFY_VERSION)
    endif
	@echo "helmify installed"
.PHONY: helmify

# Install e2e if not exist
e2e:
    ifeq (${E2E},)
		@echo "e2e not exist, installing it..."
		go install github.com/apache/skywalking-infra-e2e/cmd/e2e@$(E2E_VERSION)
    endif
	@echo "e2e installed"
.PHONY: e2e

lint: golint
.PHONY: lint

# Run go fmt against code
gofmt:
	go fmt main.go
	go fmt $(modules)
.PHONY: gofmt

fmt: gofmt
.PHONY: fmt

# Run go vet against code
vet:
	go vet main.go
	go vet $(modules)
.PHONY: vet

# Generate code
generate: controller-gen
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./..."
.PHONY: generate

# Vendor modules for code-generate
vendor:
	go mod tidy
	go mod vendor
.PHONY: vendor

# Build the docker image
docker-build:
	docker build . -t ${IMG}

# Push the docker image
docker-push:
	docker push ${IMG}

# find or download controller-gen
# download controller-gen if necessary
controller-gen:
ifeq (, $(shell which controller-gen))
	@{ \
	set -e ;\
	CONTROLLER_GEN_TMP_DIR=$$(mktemp -d) ;\
	cd $$CONTROLLER_GEN_TMP_DIR ;\
	go mod init tmp ;\
	go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION) ;\
	rm -rf $$CONTROLLER_GEN_TMP_DIR ;\
	}
CONTROLLER_GEN=$(GOBIN)/controller-gen
else
CONTROLLER_GEN=$(shell which controller-gen)
endif

kustomize:
ifeq (, $(shell which kustomize))
	@{ \
	set -e ;\
	KUSTOMIZE_GEN_TMP_DIR=$$(mktemp -d) ;\
	cd $$KUSTOMIZE_GEN_TMP_DIR ;\
	go mod init tmp ;\
	go install sigs.k8s.io/kustomize/kustomize/$(KUSTOMIZE_VERSION) ;\
	rm -rf $$KUSTOMIZE_GEN_TMP_DIR ;\
	}
KUSTOMIZE=$(GOBIN)/kustomize
else
KUSTOMIZE=$(shell which kustomize)
endif

# Generate bundle manifests and metadata, then validate generated files.
.PHONY: bundle
bundle: manifests kustomize
	operator-sdk generate kustomize manifests -q
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)
	$(KUSTOMIZE) build config/manifests | operator-sdk generate bundle -q --overwrite --version $(VERSION) $(BUNDLE_METADATA_OPTS)
	operator-sdk bundle validate ./bundle

# Build the bundle image.
.PHONY: bundle-build
bundle-build:
	docker build -f bundle.Dockerfile -t $(BUNDLE_IMG) .

# Build all bundle operator
.PHONY: generate-helm-chart
generate-helm-chart: helmify
	cd ../charts && $(KUSTOMIZE) build ../k8s/config/default | ${GOBIN}/helmify vineyard-operator && \
	sed -i 's/\/var\/run\/vineyard-kubernetes\/{{.Namespace}}\/{{.Name}}/\/var\/run\/vineyard-kubernetes\/{{ \"{{.Namespace}}\/{{.Name}}\" }}/g' \
	vineyard-operator/templates/vineyardd-crd.yaml
