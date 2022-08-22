#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

# install apache-arrow
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt update
apt install -y libarrow-dev=6.0.1-1 libparquet-dev=6.0.1-1 libarrow-python-dev=6.0.1-1

# install pyarrow from scratch
sudo pip3 install --no-binary pyarrow pyarrow==6.0.1

# apt-get cleanup
apt-get autoclean
rm -rf ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb

