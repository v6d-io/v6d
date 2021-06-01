#!/bin/bash

set -x
set -e
set -o pipefail

ROOT=$(dirname "${BASH_SOURCE[0]}")/..

helm install vineyard $ROOT/vineyard/ -n vineyard-system \
    --set image.repository="registry-vpc.cn-hongkong.aliyuncs.com/libvineyard/vineyardd" \
    --set image.tagPrefix=ubuntu \
    --set serviceAccountName=vineyard-server \
    --set env[0].name="VINEYARD_SYNC_CRDS",env[0].value="1" \
    --set vineyard.sharedMemorySize=8Gi

set +x
set +e
set +o pipefail
