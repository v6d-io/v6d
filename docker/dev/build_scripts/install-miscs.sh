#!/bin/bash

# install python packages for codegen, and io adaptors
sudo pip3 install -U "Pygments>=2.4.1"
sudo pip3 install -r build_scripts/requirements.txt

# install clang-format
sudo curl -L https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-22538c65/clang-format-8_linux-amd64 --output /usr/bin/clang-format
sudo chmod +x /usr/bin/clang-format

