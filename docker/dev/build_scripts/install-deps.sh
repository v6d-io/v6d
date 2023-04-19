#!/bin/bash

set -ex
set -o pipefail

export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

# install deps using apt-get
apt-get update -y
apt-get install -y sudo

apt-get install -y \
  ca-certificates \
  ccache \
  cmake \
  curl \
  doxygen \
  fuse3 \
  git \
  libboost-all-dev \
  libcurl4-openssl-dev \
  libfuse3-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libgmock-dev \
  libgrpc-dev \
  libgrpc++-dev \
  libkrb5-dev \
  libmpich-dev \
  libprotobuf-dev \
  librdkafka-dev \
  libgsasl7-dev \
  librdkafka-dev \
  libssl-dev \
  libunwind-dev \
  libxml2-dev \
  libz-dev \
  lsb-release \
  pandoc \
  protobuf-compiler-grpc \
  python3-pip \
  uuid-dev \
  vim \
  wget

# apt-get cleanup
apt-get autoclean
rm -rf ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb

