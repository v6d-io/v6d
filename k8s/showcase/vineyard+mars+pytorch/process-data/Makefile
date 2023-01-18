REGISTRY := docker.pkg.github.com/v6d-io/v6d
TAG      := latest

all: build-mars-process-data

build-mars-process-data:
	docker build . -f Dockerfile -t $(REGISTRY)/mars-process-data:$(TAG)
.PHONY: build-process-data

docker-push: build-mars-process-data
	docker push $(REGISTRY)/mars-process-data:$(TAG)
.PHONY: docker-push