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

# the phony is to change the image's registry to the default registry(ghcr.io/v6d-io/v6d)
# as it's hard-coded in the vineyard-operator, such as assembly、repartition、backup、recover etc.
tag-%: %
	docker tag $(REGISTRY)/$< $(DEFAULT_REGISTRY)/$<
.PHONY: tag-%

load-%: %
	kind load docker-image --name $(KIND) $(DEFAULT_REGISTRY)/$<
.PHONY: load-%

############# assembly testing ################################################

# build and tag local assembly image, as it's hard-coded in the vineyard operator templates
build-assembly-job: local-assembly tag-local-assembly load-local-assembly

local-assembly-images: local-assembly assembly-job1 assembly-job2
publish-local-assembly-images: push-local-assembly push-assembly-job1 push-assembly-job2
e2e-test-assembly: build-assembly-job push-assembly-job1 push-assembly-job2

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

############# distributed assembly testing ####################################

# build and tag distributed assembly image, as it's hard-coded in the vineyard operator templates
build-distributed-assembly-job: distributed-assembly tag-distributed-assembly load-distributed-assembly

distributed-assembly-images: distributed-assembly distributed-assembly-job1 distributed-assembly-job2 distributed-assembly-job3
publish-distributed-assembly-images: push-distributed-assembly push-distributed-assembly-job1 push-distributed-assembly-job2 push-distributed-assembly-job3
e2e-test-distributed-assembly: build-distributed-assembly-job push-distributed-assembly-job1 push-distributed-assembly-job2 push-distributed-assembly-job3

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

############# spill testing ###################################################

############# serialize testing ###################################################

serialize-images: serialize-job
publish-serialize-images: push-serialize-job
e2e-test-serialize: push-serialize-job

serialize-job:
	docker build . -f Dockerfile \
		--build-arg APP=serialize-demo/serialize.py \
		-t $(REGISTRY)/serialize-job:$(TAG)
.PHONY: serialize-job

############# failover testing ################################################

# build and tag backup image, as it's hard-coded in the vineyard operator templates
build-backup-job: backup-job tag-backup-job load-backup-job
# build and tag recover image, as it's hard-coded in the vineyard operator templates
build-recover-job: recover-job tag-recover-job load-recover-job

failover-images: backup-job recover-job build-local-object build-distributed-object-step1 build-distributed-object-step2 get-local-object get-distributed-object
publish-failover-images: push-backup-job push-recover-job push-build-local-object push-build-distributed-object-step1 push-build-distributed-object-step2 push-get-local-object push-get-distributed-object
e2e-test-failover: build-backup-job build-recover-job push-build-local-object push-build-distributed-object-step1 push-build-distributed-object-step2 push-get-local-object push-get-distributed-object

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

############# repartition testing #############################################

# build and tag dask repartition image, as it's hard-coded in the vineyard operator templates
build-repartition-job: dask-repartition tag-dask-repartition load-dask-repartition

repartition-images: dask-repartition dask-repartition-job1 dask-repartition-job2 dask-worker-with-vineyard
publish-repartition-images: push-dask-repartition push-dask-repartition-job1 push-dask-repartition-job2 push-dask-worker-with-vineyard
e2e-test-repartition: build-repartition-job push-dask-repartition-job1 push-dask-repartition-job2 push-dask-worker-with-vineyard

dask-repartition:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=dask-repartition.py \
		--build-arg PYPI_DEPENDENCIES="dask distributed vineyard-dask" \
		-t $(REGISTRY)/dask-repartition:$(TAG)
.PHONY: dask-repartition

dask-repartition-job1:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=job1.py \
		--build-arg PYPI_DEPENDENCIES="dask distributed lz4 vineyard-dask" \
		-t $(REGISTRY)/dask-repartition-job1:$(TAG)
.PHONY: dask-repartition-job1

dask-repartition-job2:
	docker build repartition-demo/ -f Dockerfile \
		--build-arg APP=job2.py \
		--build-arg PYPI_DEPENDENCIES="dask distributed lz4 vineyard-dask" \
		-t $(REGISTRY)/dask-repartition-job2:$(TAG)
.PHONY: dask-repartition-job2

dask-worker-with-vineyard:
	docker build repartition-demo/ -f repartition-demo/Dockerfile.dask-worker-with-vineyard \
		-t $(REGISTRY)/dask-worker-with-vineyard:$(TAG)
.PHONY: dask-worker-with-vineyard

############# sidecar testing #################################################

sidecar-images: sidecar-job
publish-sidecar-images: push-sidecar-job
e2e-test-sidecar: push-sidecar-job

sidecar-job:
	docker build sidecar-demo/ -f Dockerfile \
		--build-arg APP=job.py \
		-t $(REGISTRY)/sidecar-job:$(TAG)
.PHONY: sidecar-job

############# workflow testing ################################################

workflow-images: workflow-job1 workflow-job2
publish-workflow-images: push-workflow-job1 push-workflow-job2
e2e-test-workflow: push-workflow-job1 push-workflow-job2

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