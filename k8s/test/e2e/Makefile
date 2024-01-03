E2E_DIR := $(shell pwd)
TEST_DIR := ${E2E_DIR}/..
K8S_DIR := ${E2E_DIR}/../..
CMD_DIR := ${K8S_DIR}/cmd
ROOT_DIR := ${E2E_DIR}/../../..

ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# setup the registry
DEFAULT_REGISTRY := ghcr.io/v6d-io/v6d
ifeq ($(REGISTRY),)
    REGISTRY := $(DEFAULT_REGISTRY)
endif

TAG ?= latest
KIND ?= kind

push-%: %
	docker push $(REGISTRY)/$<
.PHONY: push-%

# update the docker image for vineyardd & vineyard-operator inside the kind cluster
load-vineyardd-image:
	@docker tag vineyardcloudnative/vineyardd:latest localhost:5001/vineyardd:latest
	@docker push localhost:5001/vineyardd:latest
.PHONY: load-vineyardd-image

load-vineyard-operator-image:
	@docker tag vineyardcloudnative/vineyard-operator:latest localhost:5001/vineyard-operator:latest
	@docker push localhost:5001/vineyard-operator:latest
.PHONY: load-vineyard-operator-image

# Prepare tools for e2e test
prepare-e2e-test:
	@echo "Install e2e test dependencies..."
	@make -C ${K8S_DIR} kustomize
	@make -C ${K8S_DIR} e2e
	@sudo bash ${TEST_DIR}/hack/prepare-e2e.sh
	@echo "Build vineyard operator image..."
	@make -C ${K8S_DIR} docker-build
.PHONY: prepare-e2e-test

# Build base images for e2e test
build-base-images:
	@echo "Building the python dev image"
	@make -C ${ROOT_DIR}/docker build-python-dev
	@docker tag ghcr.io/v6d-io/v6d/vineyard-python-dev:latest_x86_64 ghcr.io/v6d-io/v6d/vineyard-python-dev:latest
	@echo "Building the vineyardd image"
	@make -C ${ROOT_DIR}/docker vineyardd
	@docker tag ghcr.io/v6d-io/v6d/vineyardd:alpine-latest_x86_64 vineyardcloudnative/vineyardd:latest

# Build a 4-nodes(1 master and 3 workers) kind cluster with local registry
build-local-cluster:
	@touch /tmp/e2e-k8s.config
	@echo "the kubeconfig path is /tmp/e2e-k8s.config"
	@echo "Creating the kind cluster with local registry"
	@bash ${TEST_DIR}/hack/build-kind-cluster-with-local-registry.sh
.PHONY: build-local-cluster

# Delete the kind cluster with local registry
delete-local-cluster:
	@rm /tmp/e2e-k8s.config
	@echo "Deleting the kind cluster with local registry"
	@bash ${TEST_DIR}/hack/delete-kind-cluster-with-local-registry.sh
	# the next step is recover the default vineyard operator image(vineyardcloudnative/vineyard-operator)
	@make -C ${K8S_DIR} recover-registry
.PHONY: delete-local-cluster

# install the vineyard-operator and wait for ready
install-vineyard-operator: load-vineyard-operator-image
	@echo "Installing vineyard-operator..."
	@export IMG=localhost:5001/vineyard-operator:latest && make -C ${K8S_DIR} update-registry
	@go run ${CMD_DIR}/main.go deploy operator --kubeconfig /tmp/e2e-k8s.config
	@echo "Vineyard-Operator Ready"

# install the vineyardd and wait for ready
install-vineyardd: load-vineyardd-image
	@echo "Installing vineyard cluster..."
	@go run ${CMD_DIR}/main.go deploy vineyardd -f ${TEST_DIR}/e2e/vineyardd.yaml --kubeconfig /tmp/e2e-k8s.config
	@echo "Vineyard cluster Ready"

# install the vineyard cluster (including several components: vineyard-operator and vineyardd)
install-vineyard-cluster: build-local-cluster
	@make -C ${E2E_DIR} install-vineyard-operator
	@make -C ${E2E_DIR} install-vineyardd

############# local assembly e2e testing ################################################

local-assembly-images: local-assembly assembly-job1 assembly-job2
publish-local-assembly-images: push-local-assembly push-assembly-job1 push-assembly-job2

local-assembly:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=assembly-local.py \
		-t $(REGISTRY)/local-assembly:$(TAG)
.PHONY: local-assembly

assembly-job1:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=assembly-job1.py \
		-t $(REGISTRY)/assembly-job1:$(TAG)
.PHONY: assembly-job1

assembly-job2:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=assembly-job2.py \
		-t $(REGISTRY)/assembly-job2:$(TAG)
.PHONY: assembly-job2

e2e-tests-assembly-local: prepare-e2e-test install-vineyard-cluster
	@echo "Running local assembly e2e test..."
	@make -C ${K8S_DIR} e2e
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/assembly/local-assembly-e2e.yaml
	@echo "local assembly e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-assembly-local

############# distributed assembly testing ####################################

distributed-assembly-images: distributed-assembly distributed-assembly-job1 distributed-assembly-job2 distributed-assembly-job3
publish-distributed-assembly-images: push-local-assembly push-distributed-assembly push-distributed-assembly-job1 push-distributed-assembly-job2 push-distributed-assembly-job3

distributed-assembly:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=assembly-distributed.py \
		-t $(REGISTRY)/distributed-assembly:$(TAG)
.PHONY: distributed-assembly

distributed-assembly-job1:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=distributed-job1.py \
		-t $(REGISTRY)/distributed-assembly-job1:$(TAG)
.PHONY: distributed-assembly-job1

distributed-assembly-job2:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=distributed-job2.py \
		-t $(REGISTRY)/distributed-assembly-job2:$(TAG)
.PHONY: distributed-assembly-job2

distributed-assembly-job3:
	docker build assembly-demo/ -f Dockerfile \
		--build-arg APP=distributed-job3.py \
		-t $(REGISTRY)/distributed-assembly-job3:$(TAG)
.PHONY: build-distributed-assembly-job3

e2e-tests-assembly-distributed: prepare-e2e-test install-vineyard-cluster
	@echo "Running distributed assembly e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/assembly/distributed-assembly-e2e.yaml
	@echo "distributed assembly e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-assembly-distributed

############# autogenerated helm chart testing ###################################################

e2e-tests-autogenerated-helm-chart: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running autogenerated helm chart e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/autogenerated-helm-chart/e2e.yaml
	@echo "autogenerated helm chart e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-autogenerated-helm-chart

############# spill testing ###################################################

e2e-tests-spill: prepare-e2e-test build-local-cluster load-vineyardd-image
	@make -C ${E2E_DIR} install-vineyard-operator
	@echo "Running spill e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/spill/e2e.yaml
	@echo "spill e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-spill

############# serialize testing ###################################################

serialize-images: serialize-job
publish-serialize-images: push-serialize-job

serialize-job:
	docker build . -f Dockerfile \
		--build-arg APP=serialize-demo/serialize.py \
		-t $(REGISTRY)/serialize-job:$(TAG)
.PHONY: serialize-job

e2e-tests-serialize: prepare-e2e-test install-vineyard-cluster
	@echo "Running serialize e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/serialize/e2e.yaml
	@echo "serialize e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-serialize

############# failover testing ################################################

failover-images: backup-job recover-job build-local-object build-distributed-object-step1 build-distributed-object-step2 get-local-object get-distributed-object
publish-failover-images: push-backup-job push-recover-job push-build-local-object push-build-distributed-object-step1 push-build-distributed-object-step2 push-get-local-object push-get-distributed-object

backup-job:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=backup.py \
		-t $(REGISTRY)/backup-job:$(TAG)
.PHONY: backup-job

recover-job:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=recover.py \
		-t $(REGISTRY)/recover-job:$(TAG)
.PHONY: recover-job

build-local-object:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=build-local-object.py \
		-t $(REGISTRY)/build-local-object:$(TAG)
.PHONY: build-local-object

build-distributed-object-step1:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=build-distributed-object-step1.py \
		-t $(REGISTRY)/build-distributed-object-step1:$(TAG)
.PHONY: build-distributed-object-step1

build-distributed-object-step2:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=build-distributed-object-step2.py \
		-t $(REGISTRY)/build-distributed-object-step2:$(TAG)
.PHONY: build-distributed-object-step2

get-local-object:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=get-local-object.py \
		-t $(REGISTRY)/get-local-object:$(TAG)
.PHONY: get-local-object

get-distributed-object:
	docker build failover-demo/ -f Dockerfile \
		--build-arg APP=get-distributed-object.py \
		-t $(REGISTRY)/get-distributed-object:$(TAG)
.PHONY: get-distributed-object

e2e-tests-failover: prepare-e2e-test install-vineyard-cluster
	@echo "Running failover e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/failover/e2e.yaml
	@echo "failover e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-failover

############# dask repartition testing #############################################

repartition-images: dask-repartition dask-repartition-job1 dask-repartition-job2 dask-worker-with-vineyard
publish-repartition-images: push-dask-repartition push-dask-repartition-job1 push-dask-repartition-job2 push-dask-worker-with-vineyard

# install a lower version of pandas to avoid the conflict with dask
dask-repartition:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=dask-repartition.py \
		--build-arg PYPI_DEPENDENCIES="msgpack==1.0.4 numpy==1.23.2 pandas==1.4.3 dask==2022.8.1 distributed==2022.8.1 lz4==4.0.0 cloudpickle==2.1.0 vineyard-dask" \
		-t $(REGISTRY)/dask-repartition:$(TAG)
.PHONY: dask-repartition

dask-repartition-job1:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=job1.py \
		--build-arg PYPI_DEPENDENCIES="msgpack==1.0.4 numpy==1.23.2 pandas==1.4.3 dask==2022.8.1 distributed==2022.8.1 lz4==4.0.0 cloudpickle==2.1.0 vineyard-dask" \
		-t $(REGISTRY)/dask-repartition-job1:$(TAG)
.PHONY: dask-repartition-job1

dask-repartition-job2:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=job2.py \
		--build-arg PYPI_DEPENDENCIES="msgpack==1.0.4 numpy==1.23.2 pandas==1.4.3 dask==2022.8.1 distributed==2022.8.1 lz4==4.0.0 cloudpickle==2.1.0 vineyard-dask" \
		-t $(REGISTRY)/dask-repartition-job2:$(TAG)
.PHONY: dask-repartition-job2

dask-worker-with-vineyard:
	docker build repartition-demo/ -f repartition-demo/Dockerfile.dask-worker-with-vineyard \
		-t $(REGISTRY)/dask-worker-with-vineyard:$(TAG)
.PHONY: dask-worker-with-vineyard

e2e-tests-repartition-dask: prepare-e2e-test install-vineyard-cluster
	@echo "Running dask repartition e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/repartition/dask-repartition-e2e.yaml
	@echo "repartition dask e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-repartition-dask

############# sidecar testing #################################################

sidecar-images: sidecar-job
publish-sidecar-images: push-sidecar-job

sidecar-job:
	docker build sidecar-demo/ -f Dockerfile \
		--build-arg APP=job.py \
		-t $(REGISTRY)/sidecar-job:$(TAG)
.PHONY: sidecar-job

e2e-tests-sidecar: prepare-e2e-test build-local-cluster load-vineyardd-image
	@make -C ${E2E_DIR} install-vineyard-operator
	@echo "Running sidecar e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/sidecar/e2e.yaml
	@echo "sidecar e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-sidecar

############# workflow testing ################################################

workflow-images: workflow-job1 workflow-job2
publish-workflow-images: push-workflow-job1 push-workflow-job2

workflow-job1:
	docker build workflow-demo/ -f Dockerfile \
		--build-arg APP=job1.py \
		-t $(REGISTRY)/workflow-job1:$(TAG)
.PHONY: workflow-job1

workflow-job2:
	docker build workflow-demo/ -f Dockerfile \
		--build-arg APP=job2.py \
		-t $(REGISTRY)/workflow-job2:$(TAG)
.PHONY: workflow-job2

e2e-tests-workflow: prepare-e2e-test install-vineyard-cluster
	@echo "Running workflow e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/workflow/e2e.yaml
	@echo "workflow e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-workflow

############# schedule workflow testing ################################################

e2e-tests-schedule-workflow: prepare-e2e-test install-vineyard-cluster
	@echo "Running schedule workflow e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/schedule-workflow/e2e.yaml
	@echo "schedule workflow e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-schedule-workflow

############# schedule workflow without CRD testing ################################################

e2e-tests-schedule-workflow-without-crd: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running schedule workflow without CRD e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/schedule-workflow-without-crd/e2e.yaml
	@echo "schedule workflow e2e without CRD test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-schedule-workflow-without-crd

############# mars showcase testing ################################################

e2e-tests-mars-examples: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running mars showcase e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/mars-examples/e2e.yaml
	@echo "mars showcase e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-mars-examples

############# vineyardctl testing ################################################

e2e-tests-vineyardctl: prepare-e2e-test build-local-cluster load-vineyardd-image
	@make -C ${E2E_DIR} install-vineyard-operator
	@echo "Running vineyardctl e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/vineyardctl/e2e.yaml
	@echo "vineyardctl e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-vineyardctl

############# backup and recover testing ################################################

e2e-tests-deploy-raw-backup-and-recover: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running vineyardctl backup and recover e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/deploy-raw-backup-and-recover/e2e.yaml
	@echo "vineyardctl backup and recover e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-deploy-raw-backup-and-recover

############# schedule workload testing ################################################

e2e-tests-schedule-workload: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running schedule workload e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/schedule-workload/e2e.yaml
	@echo "schedule workload e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-schedule-workload

############# airflow integration testing ################################################

e2e-tests-airflow-integration: prepare-e2e-test build-local-cluster load-vineyardd-image
	@echo "Running airflow integration e2e test..."
	@cd ${ROOT_DIR} && ${GOBIN}/e2e run --config=${E2E_DIR}/airflow-integration/e2e.yaml
	@echo "airflow integration e2e test passed."
	@make delete-local-cluster
.PHONY: e2e-tests-airflow-integration

############# unit testing ################################################
unittest: prepare-e2e-test build-local-cluster load-vineyardd-image
	@make -C ${E2E_DIR} install-vineyard-operator
	@echo "Running vineyardctl unit test.." 
# run all unit test except deploy and scheduler in parallel
# dsiable vineyard csi driver test here
	@export KUBECONFIG=/tmp/e2e-k8s.config && cd ${CMD_DIR}/commands && go test `go list ./... | grep -Ev "deploy|schedule|csi"`
# deploy test
	@export KUBECONFIG="/tmp/e2e-k8s.config" && cd ${CMD_DIR}/commands/deploy && \
		go test -run first && go test -run second && go test -run third
# schedule test
	@export KUBECONFIG="/tmp/e2e-k8s.config" && cd ${CMD_DIR}/commands/schedule && \
		go test -run ./
	@echo "vineyardctl unit test passed."
	@make delete-local-cluster
.PHONY: unittest

# publish all images to ghcr registry
ALL-IMAGES ?= \
publish-local-assembly-images \
publish-distributed-assembly-images \
publish-serialize-images \
publish-failover-images \
publish-repartition-images \
publish-sidecar-images \
publish-workflow-images

publish-all-images: $(ALL-IMAGES)
.PHONY: publish-all-images