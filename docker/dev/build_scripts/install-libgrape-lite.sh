#!/bin/bash

# install libgrape-lite
pushd /tmp
git clone https://github.com/alibaba/libgrape-lite.git
cd libgrape-lite
mkdir build
cd build
cmake ..
make -j`nproc`
sudo make install
popd
rm -rf /tmp/libgrape-lite

