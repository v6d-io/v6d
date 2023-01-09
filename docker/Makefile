BUILDER_REGISTRY	:= ghcr.io/v6d-io/v6d
BUILDER_IMAGE		:= vineyardd-alpine-builder
BUILDER_TAG 		:= builder-latest

ALPINE_REGISTRY		:= ghcr.io/v6d-io/v6d
ALPINE_IMAGE		:= vineyardd
ALPINE_TAG			:= alpine-latest

PYTHON_DEV_REGISTRY	:= ghcr.io/v6d-io/v6d
PYTHON_DEV_IMAGE	:= vineyard-python-dev
PYTHON_DEV_TAG		:= latest

all: builder docker-build docker-push

builder: 
	docker build ./vineyardd -f ./vineyardd/Dockerfile.alpine-builder -t $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_TAG)
.PHONY: builder

docker-build: builder
	docker build .. -f Dockerfile.vineyardd -t $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_TAG)
.PHONY: docker-build

build-python-dev: 
	docker build .. -f Dockerfile.vineyard-python-dev -t $(PYTHON_DEV_REGISTRY)/$(PYTHON_DEV_IMAGE):$(PYTHON_DEV_TAG)
.PHONY: build-python-dev

docker-push: docker-build
	docker push $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_TAG)
.PHONY: docker-push
