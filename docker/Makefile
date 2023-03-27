REGISTRY 					:= ghcr.io/v6d-io/v6d

# can be: x86_64, aarch64
PLATFORM					:= x86_64
ARCH 						:= $(subst x86_64,amd64,$(subst aarch64,arm64,$(PLATFORM)))

BUILDER_REGISTRY			:= $(REGISTRY)
BUILDER_IMAGE				:= vineyardd-alpine-builder
BUILDER_MANIFEST_TAG 		:= builder-latest
BUILDER_TAG 				:= $(BUILDER_MANIFEST_TAG)_$(PLATFORM)

ALPINE_REGISTRY				:= $(REGISTRY)
ALPINE_IMAGE				:= vineyardd
ALPINE_VERSION 				:= latest
ALPINE_MANIFEST_TAG			:= alpine-$(ALPINE_VERSION)
ALPINE_TAG					:= $(ALPINE_MANIFEST_TAG)_$(PLATFORM)

WHEEL_BUILDER_REGISTRY		:= $(REGISTRY)
WHEEL_BUILDER_IMAGE			:= vineyard-manylinux2014
WHEEL_BUILDER_MANIFEST_TAG 	:= 20230316
WHEEL_BUILDER_TAG 			:= $(WHEEL_BUILDER_MANIFEST_TAG)_$(PLATFORM)

WHEEL_PYTHON				:= cp311-cp311
WHEEL_VERSION				:= latest

WHEEL_REGISTRY				:= $(REGISTRY)
WHEEL_IMAGE					:= vineyard-wheel
WHEEL_MANIFEST_TAG 			:= $(WHEEL_VERSION)_$(WHEEL_PYTHON)
WHEEL_TAG 					:= $(WHEEL_MANIFEST_TAG)_$(PLATFORM)

PYTHON_DEV_REGISTRY			:= $(REGISTRY)
PYTHON_DEV_IMAGE			:= vineyard-python-dev
PYTHON_DEV_TAG				:= latest_$(PLATFORM)

all: docker-build docker-push

# builder for building standalone vineyardd binary
builder:
	docker build ./vineyardd/ \
		-f ./vineyardd/Dockerfile.alpine-builder \
		-t $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_TAG) \
		--build-arg PLATFORM=$(PLATFORM) \
		--build-arg ARCH=$(ARCH)
.PHONY: builder

builder-manifest:
	docker manifest create $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_MANIFEST_TAG) \
		--amend $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_MANIFEST_TAG)_x86_64 \
		--amend $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_MANIFEST_TAG)_aarch64
.PHONY: builder-manifest

# building standalone vineyardd
vineyardd:
	docker buildx build ../ \
		--file ./Dockerfile.vineyardd \
		-t ghcr.io/v6d-io/v6d/vineyardd:$(ALPINE_TAG) \
		--load \
		--platform linux/$(ARCH)
.PHONY: vineyardd

vineyardd-manifest:
	docker manifest create $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_MANIFEST_TAG) \
		--amend $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_MANIFEST_TAG)_x86_64 \
		--amend $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_MANIFEST_TAG)_aarch64
.PHONY: vineyardd-manifest

# builder for building python wheel packages
wheel-builder:
	docker build ./pypa/ \
		-f ./pypa/Dockerfile.manylinux1 \
		-t $(WHEEL_BUILDER_REGISTRY)/$(WHEEL_BUILDER_IMAGE):$(WHEEL_BUILDER_TAG) \
		--build-arg PLATFORM=$(PLATFORM) \
		--build-arg ARCH=$(ARCH)
.PHONY: wheel-builder

wheel-builder-manifest:
	docker manifest create $(WHEEL_BUILDER_REGISTRY)/$(WHEEL_BUILDER_IMAGE):$(WHEEL_BUILDER_MANIFEST_TAG) \
		--amend $(WHEEL_BUILDER_REGISTRY)/$(WHEEL_BUILDER_IMAGE):$(WHEEL_BUILDER_MANIFEST_TAG)_x86_64 \
		--amend $(WHEEL_BUILDER_REGISTRY)/$(WHEEL_BUILDER_IMAGE):$(WHEEL_BUILDER_MANIFEST_TAG)_aarch64
.PHONY: wheel-builder-manifest

# building python wheel packages
python-wheel:
	docker buildx build ../ \
		-f ./pypa/Dockerfile.manylinux1-wheel \
    	-t $(WHEEL_REGISTRY)/$(WHEEL_IMAGE):$(WHEEL_TAG) \
        --build-arg BASE_VERSION=$(WHEEL_BUILDER_MANIFEST_TAG) \
        --build-arg python=$(WHEEL_PYTHON) \
		--load \
		--platform linux/$(ARCH)
.PHONY: python-wheel

# build python-dev image
build-python-dev:
	docker build ../ \
		-f ./Dockerfile.vineyard-python-dev \
		-t $(PYTHON_DEV_REGISTRY)/$(PYTHON_DEV_IMAGE):$(PYTHON_DEV_TAG)
.PHONY: build-python-dev
