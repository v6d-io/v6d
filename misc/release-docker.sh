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
