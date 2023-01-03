# Copyright 2020-2022 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

REGISTRY := ghcr.io/v6d-io/v6d
TAG      := latest

all: build-local-object build-distributed-object-step1 build-distributed-object-step2 build-get-local-object build-get-distributed-object
	
build-local-object:
	docker build . -f Dockerfile.build-local-object -t $(REGISTRY)/local-object:$(TAG)
.PHONY: build-local-object

build-distributed-object-step1:
	docker build . -f Dockerfile.build-distributed-object-step1 -t $(REGISTRY)/distributed-object-step1:$(TAG)
.PHONY: build-distributed-object-step1

build-distributed-object-step2:
	docker build . -f Dockerfile.build-distributed-object-step2 -t $(REGISTRY)/distributed-object-step2:$(TAG)
.PHONY: build-distributed-object-step2

build-get-local-object:
	docker build . -f Dockerfile.get-local-object-job -t $(REGISTRY)/get-local-object:$(TAG)
.PHONY: build-get-local-object

build-get-distributed-object:
	docker build . -f Dockerfile.get-distributed-object-job -t $(REGISTRY)/get-distributed-object:$(TAG)
.PHONY: build-get-distributed-object

docker-push: build-local-object build-distributed-object-step1 build-distributed-object-step2 build-get-local-object build-get-distributed-object
	docker push $(REGISTRY)/local-object:$(TAG)
	docker push $(REGISTRY)/distributed-object-step1:$(TAG)
	docker push $(REGISTRY)/distributed-object-step2:$(TAG)
	docker push $(REGISTRY)/get-local-object:$(TAG)
.PHONY: docker-push