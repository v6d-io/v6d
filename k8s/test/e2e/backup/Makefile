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

all: build-backup-job build-recover-job
	
build-backup-job:
	sed 's/app/backup/' Dockerfile.build | docker build -t $(REGISTRY)/backup-job:$(TAG) . -f-
.PHONY: build-backup-job

build-recover-job:
	sed 's/app/recover/' Dockerfile.build | docker build -t $(REGISTRY)/recover-job:$(TAG) . -f-
.PHONY: build-recover-job

docker-push: build-backup-job build-recover-job build-get-object-job
	docker push $(REGISTRY)/backup-job:$(TAG)
	docker push $(REGISTRY)/recover-job:$(TAG)
.PHONY: docker-push