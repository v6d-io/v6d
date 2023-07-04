#!/bin/bash

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

INSTALL_DIR=/usr/local/bin

OS=$(go env GOOS)
ARCH=$(go env GOHOSTARCH)

# install kubectl
if ! command -v kubectl &> /dev/null; then
    @echo "installing kubectl..."
    curl -LO https://dl.k8s.io/release/v1.24.0/bin/${OS}/${ARCH}/kubectl && chmod +x ./kubectl && mv ./kubectl ${INSTALL_DIR}
    if [ $? -ne 0 ]; then
        @echo "unable to install kubectl, please check"
    fi
fi

# install helm
if ! command -v helm &> /dev/null; then
    @echo "installing helm..."
    wget https://get.helm.sh/helm-v3.9.3-${OS}-${ARCH}.tar.gz -O - |\ 
    tar xz && mv ${OS}-${ARCH}/helm ${INSTALL_DIR}
    if [ $? -ne 0 ]; then
        @echo "unable to install helm, please check"
    fi
fi

# install yq
if ! command -v yq &> /dev/null; then
    @echo "installing yq..."
    wget https://github.com/mikefarah/yq/releases/download/v4.27.2/yq_${OS}_${ARCH}.tar.gz -O - |\
    tar xz && mv yq_${OS}_${ARCH} ${INSTALL_DIR}/yq
    if [ $? -ne 0 ]; then
        @echo "unable to install yq, please check."
    fi
fi

# install gomplate
if ! command -v gomplate &> /dev/null; then
    @echo "installing gomplate..."
    wget -q https://github.com/hairyhenderson/gomplate/releases/download/v3.11.3/gomplate_${OS}-${ARCH} &&  \
    chmod +x ./gomplate_${OS}-${ARCH} && mv ./gomplate_${OS}-${ARCH} ${INSTALL_DIR}/gomplate
    if [ $? -ne 0 ]; then
        @echo "unable to install gomplate, please check."
    fi
fi
