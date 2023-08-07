#!/bin/bash

set -e
set -o pipefail

base=`pwd`
version=$1

if [ "$version" = "" ]; then
    echo "./prepare-wheels.sh: requires a specific version"
    exit 1
fi

mkdir -p wheels/$version

# download wheels from github artifacts
pushd wheels/$version
curl -s https://api.github.com/repos/v6d-io/v6d/releases/tags/$version | grep "browser_download_url.*whl" | cut -d : -f 2,3 | tr -d \" | wget -qi -
popd

twine check ./wheels/$version/*.whl
for wheel in $(ls ./wheels/$version/*.whl); do
    twine upload $wheel -u __token__ -p "$PYPI_TOKEN" || true
done
