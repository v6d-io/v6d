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