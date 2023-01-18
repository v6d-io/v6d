REGISTRY := ghcr.io/v6d-io/v6d
TAG      := latest

all: build-train-data

build-train-data:
	docker build . -f Dockerfile -t $(REGISTRY)/train-data:$(TAG)
.PHONY: build-train-data

docker-push: build-mars-train-data
	docker push $(REGISTRY)/train-data:$(TAG)
.PHONY: docker-push