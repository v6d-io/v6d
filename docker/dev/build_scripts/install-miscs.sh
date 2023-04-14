#!/bin/bash

set -ex
set -o pipefail

# install python packages for codegen, and io adaptors
sudo pip3 install -U "Pygments>=2.4.1"
sudo pip3 install -r build_scripts/requirements.txt

# install clang-format
sudo curl -L https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-1d7ec53d/clang-format-11_linux-amd64 --output /usr/bin/clang-format
sudo chmod +x /usr/bin/clang-format
