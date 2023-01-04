# Copyright 2020-2023 Alibaba Group Holding Limited.
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

REGISTRY := docker.pkg.github.com/v6d-io/v6d
TAG      := latest

all: build-assembly-job1 build-assembly-job2 build-distributed-assembly-job1 build-distributed-assembly-job2 build-distributed-assembly-job3
	
build-assembly-job1:
	sed 's/app/assembly-job1/' Dockerfile.build | docker build -t $(REGISTRY)/assembly-job1:$(TAG) . -f-
.PHONY: build-assembly-job1

build-assembly-job2:
	sed 's/app/assembly-job2/' Dockerfile.build | docker build -t $(REGISTRY)/assembly-job2:$(TAG) . -f-
.PHONY: build-assembly-job2

build-distributed-assembly-job1:
	sed 's/app/distributed-job1/' Dockerfile.build | docker build -t $(REGISTRY)/distributed-assembly-job1:$(TAG) . -f-
.PHONY: build-distributed-assembly-job1

build-distributed-assembly-job2:
	sed 's/app/distributed-job2/' Dockerfile.build | docker build -t $(REGISTRY)/distributed-assembly-job2:$(TAG) . -f-
.PHONY: build-distributed-assembly-job2

build-distributed-assembly-job3:
	sed 's/app/distributed-job3/' Dockerfile.build | docker build -t $(REGISTRY)/distributed-assembly-job3:$(TAG) . -f-
.PHONY: build-distributed-assembly-job3

docker-push: build-assembly-job1 build-assembly-job2 build-distributed-assembly-job1 build-distributed-assembly-job2 build-distributed-assembly-job3
	docker push $(REGISTRY)/assembly-job1:$(TAG)
	docker push $(REGISTRY)/assembly-job2:$(TAG)
	docker push $(REGISTRY)/distributed-assembly-job1:$(TAG)
	docker push $(REGISTRY)/distributed-assembly-job2:$(TAG)
	docker push $(REGISTRY)/distributed-assembly-job3:$(TAG)
.PHONY: docker-push