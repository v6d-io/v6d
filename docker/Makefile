REGISTRY 			:= registry-vpc.cn-hongkong.aliyuncs.com
IMAGE 				:= libvineyard/vineyardd

ALPINE_REGISTRY		:= docker.pkg.github.com/v6d-io
ALPINE_IMAGE		:= v6d/vineyardd

ubuntu: ubuntu-build ubuntu-push

ubuntu-build: Dockerfile.ubuntu
	docker build .. -f Dockerfile.ubuntu -t $(REGISTRY)/$(IMAGE):ubuntu
.PHONY: ubuntu-build

ubuntu-push: Dockerfile.ubuntu
	docker push $(REGISTRY)/$(IMAGE):ubuntu
.PHONY: ubuntu-push

alpine:
	docker build .. -f Dockerfile.vineyardd -t $(ALPINE_REGISTRY)/$(ALPINE_IMAGE):alpine-latest
.PHONY: alpine
