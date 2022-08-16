ALPINE_REGISTRY		:= docker.pkg.github.com/v6d-io
ALPINE_IMAGE		:= v6d/vineyardd

docker-build:
	docker build .. -f Dockerfile.vineyardd -t $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):alpine-latest
.PHONY: docker-build
