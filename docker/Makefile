BUILDER_REGISTRY	:= docker.pkg.github.com/v6d-io
BUILDER_IMAGE		:= v6d/vineyardd-alpine-builder
BUILDER_TAG 		:= builder-latest

ALPINE_REGISTRY		:= docker.pkg.github.com/v6d-io
ALPINE_IMAGE		:= v6d/vineyardd
ALPINE_TAG			:= alpine-latest

all: builder docker-build docker-push

builder: 
	docker build ./vineyardd -f ./vineyardd/Dockerfile.alpine-builder -t $(BUILDER_REGISTRY)/$(BUILDER_IMAGE):$(BUILDER_TAG)
.PHONY: builder

docker-build: builder
	docker build .. -f Dockerfile.vineyardd -t $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_TAG)
.PHONY: docker-build

docker-push: docker-build
	docker push $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):$(ALPINE_TAG)
.PHONY: docker-push
