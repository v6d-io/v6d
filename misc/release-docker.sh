#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./prepare-docker.sh: requires a specific version"
    exit 1
fi

docker pull docker.pkg.github.com/v6d-io/v6d/vineyardd:$version
docker pull docker.pkg.github.com/v6d-io/v6d/vineyardd:latest

docker tag docker.pkg.github.com/v6d-io/v6d/vineyardd:$version vineyardcloudnative/vineyardd:$version
docker tag docker.pkg.github.com/v6d-io/v6d/vineyardd:latest vineyardcloudnative/vineyardd:latest

docker push vineyardcloudnative/vineyardd:$version
docker push vineyardcloudnative/vineyardd:latest
