#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./prepare-docker.sh: requires a specific version"
    exit 1
fi

docker pull docker.pkg.github.com/alibaba/libvineyard/vineyardd:$version
docker pull docker.pkg.github.com/alibaba/libvineyard/vineyardd:latest

docker tag docker.pkg.github.com/alibaba/libvineyard/vineyardd:$version libvineyard/vineyardd:$version
docker tag docker.pkg.github.com/alibaba/libvineyard/vineyardd:latest libvineyard/vineyardd:latest
docker tag docker.pkg.github.com/alibaba/libvineyard/vineyardd:$version quay.io/libvineyard/vineyardd:$version
docker tag docker.pkg.github.com/alibaba/libvineyard/vineyardd:latest quay.io/libvineyard/vineyardd:latest

docker push libvineyard/vineyardd:$version
docker push libvineyard/vineyardd:latest
docker push quay.io/libvineyard/vineyardd:$version
docker push quay.io/libvineyard/vineyardd:latest
