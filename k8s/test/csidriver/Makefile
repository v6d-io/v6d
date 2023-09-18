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

VERSION ?= latest

REGISTRY	:= vineyardcloudnative

VINEYARD_CSI_TEST_IMAGE ?= $(REGISTRY)/vineyard-csi-driver-test:$(VERSION)

.PHONY: csidriver-test
csidriver-test:
	cd ../../../ && \
	if docker build --help | grep -q load; then \
		docker build --load -f k8s/test/csidriver/Dockerfile . \
		-t $(VINEYARD_CSI_TEST_IMAGE); \
	else \
		docker build -f k8s/test/csidriver/Dockerfile . \
		-t $(VINEYARD_CSI_TEST_IMAGE); \
	fi && \
	docker run --rm -it --privileged=true $(VINEYARD_CSI_TEST_IMAGE)
