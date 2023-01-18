REGISTRY := ghcr.io/v6d-io/v6d
TAG      := latest

all: build-prepare-data

build-prepare-data:
	docker build . -f Dockerfile -t $(REGISTRY)/prepare-data:$(TAG)
.PHONY: build-prepare-data

docker-push: build-mars-prepare-data
	docker push $(REGISTRY)/prepare-data:$(TAG)
.PHONY: docker-push