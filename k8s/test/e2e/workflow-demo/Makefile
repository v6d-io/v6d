REGISTRY := docker.pkg.github.com/v6d-io/v6d
TAG      := latest

all: build-job1 build-job2 docker-push
	
build-job1:
	docker build . -f Dockerfile.job1 -t $(REGISTRY)/job1:$(TAG)
.PHONY: build-job1

build-job2:
	docker build . -f Dockerfile.job2 -t $(REGISTRY)/job2:$(TAG)
.PHONY: build-job2

docker-push: build-job1 build-job2
	docker push $(REGISTRY)/job1:$(TAG)
	docker push $(REGISTRY)/job2:$(TAG)
.PHONY: docker-push