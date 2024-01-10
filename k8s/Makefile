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
VERSION ?= latest
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
REGISTRY	:= vineyardcloudnative
IMG 		?= $(REGISTRY)/vineyard-operator:$(VERSION)
VINEYARD_CSI_IMAGE ?= $(REGISTRY)/vineyard-csi-driver:$(VERSION)

temp=$(shell mktemp -d)
# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

GOLINT = $(shell command -v $(GOBIN)/golangci-lint 2> /dev/null)
HELMIFY = $(shell command -v $(GOBIN)/helmify 2> /dev/null)
CRD_REF_DOCS = $(shell command -v $(GOBIN)/crd-ref-docs 2> /dev/null)
E2E = $(shell command -v $(GOBIN)/e2e 2> /dev/null)
KUSTOMIZE = $(shell command -v $(GOBIN)/kustomize 2> /dev/null)
CONTROLLER_GEN = $(shell command -v $(GOBIN)/controller-gen 2> /dev/null)

## Tool Versions
KUSTOMIZE_VERSION ?= v4@v4.5.5
CONTROLLER_TOOLS_VERSION ?= v0.11.0
GOLANGCI_LINT_VERSION ?= v1.50.0
HELMIFY_VERSION ?= v0.3.34
CRD_REF_DOCS_VERSION ?= latest
E2E_VERSION ?= 2631e76926604c4e30ca170bed916804c86980b6

all: vineyardctl

#check ci locally
.PHONY: check
check: lint e2e-test

# Build vineyardctl binary
.PHONY: vineyardctl
vineyardctl: generate fmt
	go build -a -o vineyardctl cmd/main.go

# Copy the artifacts into the python directory for bdist_wheel.
.PHONY: bdist_wheel
bdist_wheel:
	@cp vineyardctl ../python/vineyard/bdist/vineyardctl
	@strip ../python/vineyard/bdist/vineyardctl

# Run vineyardctl binary
.PHONY: run
run: generate fmt
	go run cmd/main.go

# Install CRDs into a cluster
.PHONY: install
install: manifests kustomize
	$(KUSTOMIZE) build config/crd | kubectl apply -f -

# Uninstall CRDs from a cluster
.PHONY: uninstall
uninstall: manifests kustomize
	$(KUSTOMIZE) build config/crd | kubectl delete -f -

# Deploy controller in the configured Kubernetes cluster in ~/.kube/config
.PHONY: deploy
deploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)
	$(KUSTOMIZE) build config/default | kubectl apply -f -

# Undeploy controller in the configured Kubernetes cluster in ~/.kube/config
.PHONY: undeploy
undeploy: kustomize
	kubectl -n vineyard-system delete deployment vineyard-controller-manager

# Undeploy controller and all created resources in the configured Kubernetes cluster in ~/.kube/config
.PHONY: undeploy-all
undeploy-all: kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)
	$(KUSTOMIZE) build config/default | kubectl delete -f -

# Deploy namespace, CRDs, rbac, etc. except the controller in the configured Kubernetes cluster in ~/.kube/config
.PHONY: predeploy
predeploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)

.PHONY: unpredeploy
unpredeploy: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)

# update the operator image's registry to localhost
.PHONY: update-registry
update-registry: kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)

# recover the operator image's registry to vineyardcloudnative
.PHONY: recover-registry
recover-registry: kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)

# dump the deployment configuration
.PHONY: dry-run
dry-run: manifests kustomize
	cd config/manager && $(KUSTOMIZE) edit set image controller=$(IMG)
	$(KUSTOMIZE) build config/default > controller.yaml

# Generate manifests e.g. CRD, RBAC etc.
.PHONY: manifests
manifests: controller-gen
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd:maxDescLen=0 webhook paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: sample
sample: predeploy kustomize
	$(KUSTOMIZE) build config/samples | kubectl apply -f -

.PHONY: unsample
unsample: kustomize
	$(KUSTOMIZE) build config/samples | kubectl delete -f -

# Go modules
modules	= ./apis/... \
		  ./cmd/... \
		  ./controllers/... \
		  ./pkg/...

# Run golangci-lint
.PHONY: golint
golint:
    ifeq ($(GOLINT),)
		@echo "golangci-lint not exist, installing it..."
		go install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
    endif
	@echo "golangci-lint installed"
	$(eval GOLINT = $(GOBIN)/golangci-lint)

# Install helmify if not exist
.PHONY: helmify
helmify:
    ifeq ($(HELMIFY),)
		@echo "helmify not exist, installing it..."
		go install github.com/arttor/helmify/cmd/helmify@$(HELMIFY_VERSION)
    endif
	@echo "helmify installed"
	$(eval HELMIFY = $(GOBIN)/helmify)

# Install crd-ref-docs if not exist
.PHONY: crd-ref-docs
crd-ref-docs:
    ifeq ($(CRD_REF_DOCS),)
		@echo "crd-ref-docs not exist, installing it..."
		go install github.com/elastic/crd-ref-docs@$(CRD_REF_DOCS_VERSION)
    endif
	@echo "crd-ref-docs installed"
	$(eval CRD_REF_DOCS = $(GOBIN)/crd-ref-docs)

# Install e2e if not exist
.PHONY: e2e
e2e:
    ifeq ($(E2E),)
		@echo "e2e not exist, installing it..."
		go install github.com/apache/skywalking-infra-e2e/cmd/e2e@$(E2E_VERSION) 2> /dev/null
    endif
	@echo "e2e installed"

.PHONY: lint
lint: golint
	$(GOLINT) run $(modules)

# Run go fmt against code
.PHONY: gofmt
gofmt:
	go fmt $(modules)

.PHONY: fmt
fmt: gofmt

# Run go vet against code
.PHONY: vet
vet:
	go vet $(modules)

# Generate code
.PHONY: generate
generate: controller-gen
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./..."

.PHONY: generate-crd-docs
generate-crd-docs: crd-ref-docs
	$(CRD_REF_DOCS) --source-path=apis/k8s/v1alpha1 --config=config.yaml \
	--renderer=markdown --output-path=apis/k8s/v1alpha1/README.md

# Vendor modules for code-generate
.PHONY: vendor
vendor:
	go mod tidy
	go mod vendor

# Build the docker image
.PHONY: docker-build
docker-build:
	cd .. && \
	if docker build --help | grep -q load; then \
		docker build --load -f k8s/Dockerfile . -t $(IMG); \
	else \
		docker build -f k8s/Dockerfile . -t $(IMG); \
	fi

docker-build-push-multi-arch:
	cd .. && \
	docker buildx build -f k8s/Dockerfile . -t $(IMG) --platform linux/amd64,linux/arm64 --push

# Push the docker image
.PHONY: docker-push
docker-push:
	docker push $(IMG)

# find or download controller-gen
# download controller-gen if necessary
.PHONY: controller-gen
controller-gen:
    ifeq ($(CONTROLLER_GEN),)
		@{ \
		set -e ;\
		CONTROLLER_GEN_TMP_DIR=$$(mktemp -d) ;\
		cd $$CONTROLLER_GEN_TMP_DIR ;\
		go mod init tmp ;\
		go install sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_TOOLS_VERSION) ;\
		rm -rf $$CONTROLLER_GEN_TMP_DIR ;\
		}
    endif
	$(eval CONTROLLER_GEN = $(GOBIN)/controller-gen)

.PHONY: kustomize
kustomize:
    ifeq ($(KUSTOMIZE),)
		@{ \
		set -e ;\
		KUSTOMIZE_GEN_TMP_DIR=$$(mktemp -d) ;\
		cd $$KUSTOMIZE_GEN_TMP_DIR ;\
		go mod init tmp ;\
		go install sigs.k8s.io/kustomize/kustomize/$(KUSTOMIZE_VERSION) ;\
		rm -rf $$KUSTOMIZE_GEN_TMP_DIR ;\
		}
    endif
	$(eval KUSTOMIZE = $(GOBIN)/kustomize)

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
generate-helm-chart: helmify kustomize
	cd ../charts && $(KUSTOMIZE) build ../k8s/config/default | $(HELMIFY) vineyard-operator && \
	sed -i 's/\/var\/run\/vineyard-kubernetes\/{{.Namespace}}\/{{.Name}}/\/var\/run\/vineyard-kubernetes\/{{ \"{{.Namespace}}\/{{.Name}}\" }}/g' \
	vineyard-operator/templates/vineyardd-crd.yaml && \
	sed -i '/- --verbose/a \        - --namespace\n        - {{ .Release.Namespace }}\n' vineyard-operator/templates/deployment.yaml && \
	echo 'fullnameOverride: vineyard' >> vineyard-operator/values.yaml
