#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./prepare-docker.sh: requires a specific version"
    exit 1
fi

# vineyardd
for tag in ${version} latest alpine-latest; do
    for arch in x86_64 aarch64; do
        docker pull ghcr.io/v6d-io/v6d/vineyardd:${tag}_${arch}
        docker tag ghcr.io/v6d-io/v6d/vineyardd:${tag}_${arch} vineyardcloudnative/vineyardd:${tag}_${arch}
        docker push vineyardcloudnative/vineyardd:${tag}_${arch}
    done

    docker manifest create vineyardcloudnative/vineyardd:${tag} \
        --amend vineyardcloudnative/vineyardd:${tag}_x86_64 \
        --amend vineyardcloudnative/vineyardd:${tag}_aarch64
    docker manifest push vineyardcloudnative/vineyardd:${tag}
done

# vineyard-operator
for tag in ${version} latest; do
    # see also: https://stackoverflow.com/questions/68317302/is-it-possible-to-copy-a-multi-os-image-from-one-docker-registry-to-another-on-a
    regctl image copy -v info ghcr.io/v6d-io/v6d/vineyard-operator:${tag} vineyardcloudnative/vineyard-operator:${tag}
done
