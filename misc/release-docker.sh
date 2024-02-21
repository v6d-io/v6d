#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./release-docker.sh: requires a specific version"
    exit 1
fi

# see also:
# - https://stackoverflow.com/questions/68317302/is-it-possible-to-copy-a-multi-os-image-from-one-docker-registry-to-another-on-a

# vineyardd
for tag in ${version} latest alpine-${version} alpine-latest; do
    regctl image copy -v info ghcr.io/v6d-io/v6d/vineyardd:alpine-${version} \
                              vineyardcloudnative/vineyardd:${tag}
done

# vineyard-operator
for tag in ${version} latest; do
    regctl image copy -v info ghcr.io/v6d-io/v6d/vineyard-operator:${version} \
                              vineyardcloudnative/vineyard-operator:${tag}
done

# vineyard-fluid-fuse
for tag in ${version} latest; do
    regctl image copy -v info ghcr.io/v6d-io/v6d/vineyard-fluid-fuse:${version} \
                              vineyardcloudnative/vineyard-fluid-fuse:${tag}
done
