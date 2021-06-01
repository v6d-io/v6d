#!/bin/bash

# Copyright 2020-2021 Alibaba Group Holding Limited.
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

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
CODEGEN_PKG=${CODEGEN_PKG:-$(cd "${SCRIPT_ROOT}"; ls -d -1 ./vendor/k8s.io/code-generator 2>/dev/null || echo ../code-generator)}

bash "${CODEGEN_PKG}"/generate-groups.sh \
  client,lister,informer \
  github.com/v6d-io/v6d/k8s/generated \
  github.com/v6d-io/v6d/k8s/api \
  "k8s:v1alpha1" \
  --go-header-file "${SCRIPT_ROOT}"/hack/boilerplate.go.txt -v 100 --output-base ./

# mv generated code to this package
rm -rf "${SCRIPT_ROOT}"/generated
mv "${SCRIPT_ROOT}"/github.com/v6d-io/v6d/k8s/generated "${SCRIPT_ROOT}"/
rm -rf "${SCRIPT_ROOT}"/github.com
