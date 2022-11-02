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

all: build-dask-repartition-job1 build-dask-repartition-job2 build-dask-worker-with-vineyard
	
build-dask-repartition-job1:
	docker build . -f Dockerfile.job1 -t $(REGISTRY)/dask-repartition-job1:$(TAG)
.PHONY: build-dask-repartition-job1

build-dask-repartition-job2:
	docker build . -f Dockerfile.job2 -t $(REGISTRY)/dask-repartition-job2:$(TAG)
.PHONY: build-dask-repartition-job2

build-dask-worker-with-vineyard:
	docker build . -f Dockerfile.dask-worker-with-vineyard -t $(REGISTRY)/dask-worker-with-vineyard:$(TAG)

docker-push: build-dask-repartition-job1 build-dask-repartition-job2 build-dask-worker-with-vineyard
	docker push $(REGISTRY)/dask-repartition-job1:$(TAG)
	docker push $(REGISTRY)/dask-repartition-job2:$(TAG)
	docker push $(REGISTRY)/dask-worker-with-vineyard:$(TAG)
.PHONY: docker-push