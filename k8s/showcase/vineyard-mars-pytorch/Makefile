# setup the registry
DEFAULT_REGISTRY := vineyardcloudnative
ifeq ($(REGISTRY),)
    REGISTRY := $(DEFAULT_REGISTRY)
endif
TAG      := latest

all: build-prepare-data build-mars-process-data build-train-data

build-prepare-data:
	docker build prepare-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=prepare-data.py \
		-t $(REGISTRY)/prepare-data:$(TAG)
.PHONY: build-prepare-data

build-mars-process-data:
	docker build process-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=process-data.py \
		--build-arg PYPI_DEPENDENCIES="pymars==0.10.0 requests urllib3" \
		-t $(REGISTRY)/mars-process-data:$(TAG)
.PHONY: build-process-data

build-train-data:
	docker build train-data/ -f ../../test/e2e/Dockerfile \
		--build-arg APP=train-data.py \
		--build-arg PYPI_DEPENDENCIES="torch==1.13.1+cpu" \
		--build-arg EXTRA_DEPENDENCIES="https://download.pytorch.org/whl/cpu" \
		-t $(REGISTRY)/train-data:$(TAG)
.PHONY: build-train-data

# TODO: build the image from the next mars released image.
# At the time, we build the image `ghcr.io/v6d-io/v6d/mars-with-vineyard:v0.10.0` 
# from mars project and install vineyard there. The main process is as follows.
# git clone https://github.com/mars-project/mars.git
# Add `pip3 install vineyard` in mars/deploy/kubernetes/docker/Dockerfile
# Also, we should specify the pandas's version as 1.5.2.
# Then run `bash bin/kube-image-tool.sh -o "ghcr.io/v6d-io/v6d/mars-with-vineyard" -t "v0.10.0" build`
# Next tag the built image `ghcr.io/v6d-io/v6d/mars-with-vineyard/mars:v0.10.0` to `ghcr.io/v6d-io/v6d/mars-with-vineyard:v0.10.0`
# Finally push it

docker-push: build-prepare-data build-mars-process-data build-train-data
	docker push $(REGISTRY)/prepare-data:$(TAG)
	docker push $(REGISTRY)/mars-process-data:$(TAG)
	docker push $(REGISTRY)/train-data:$(TAG)
.PHONY: docker-push