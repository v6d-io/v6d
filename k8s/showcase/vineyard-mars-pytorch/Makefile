REGISTRY := ghcr.io/v6d-io/v6d
TAG      := latest

all: build-prepare-data build-mars-process-data build-train-data

build-prepare-data:
	docker build prepare-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=prepare-data.py \
		--build-arg BASE_IMAGE=python:3.10 \
		--build-arg PYPI_DEPENDENCIES="vineyard==0.11.7" \
		-t $(REGISTRY)/prepare-data:$(TAG)
.PHONY: build-prepare-data

build-mars-process-data:
	docker build process-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=process-data.py \
		--build-arg BASE_IMAGE=python:3.10 \
		--build-arg PYPI_DEPENDENCIES="vineyard==0.11.7 pymars==0.10.0 requests urllib3" \
		-t $(REGISTRY)/mars-process-data:$(TAG)
.PHONY: build-process-data

build-train-data:
	docker build train-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=train-data.py \
		--build-arg BASE_IMAGE=python:3.10 \
		--build-arg PYPI_DEPENDENCIES="vineyard==0.11.7 torch" \
		-t $(REGISTRY)/train-data:$(TAG)
.PHONY: build-train-data

docker-push: build-prepare-data build-mars-process-data build-train-data
	docker push $(REGISTRY)/prepare-data:$(TAG)
	docker push $(REGISTRY)/mars-process-data:$(TAG)
	docker push $(REGISTRY)/train-data:$(TAG)
.PHONY: docker-push