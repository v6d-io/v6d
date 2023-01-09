#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./prepare-docker.sh: requires a specific version"
    exit 1
fi

docker pull ghcr.io/v6d-io/v6d/vineyardd:$version
docker pull ghcr.io/v6d-io/v6d/vineyardd:latest
docker pull ghcr.io/v6d-io/v6d/vineyard-operator:nightly

docker tag ghcr.io/v6d-io/v6d/vineyardd:$version vineyardcloudnative/vineyardd:$version
docker tag ghcr.io/v6d-io/v6d/vineyardd:latest vineyardcloudnative/vineyardd:latest
docker tag ghcr.io/v6d-io/v6d/vineyard-operator:nightly vineyardcloudnative/vineyard-operator:nightly

docker push vineyardcloudnative/vineyardd:$version
docker push vineyardcloudnative/vineyardd:latest
docker push vineyardcloudnative/vineyard-operator:nightly
