#!/usr/bin/env bash

# git-archive-submodules - A script to produce an archive tar.gz file of the a git module including all git submodules
# based on https://ttboj.wordpress.com/2015/07/23/git-archive-with-submodules-and-tar-magic/

# This script is modified from https://github.com/nzanepro/git-archive-submodules, which originally
# has the following license header:
#
#   MIT License
#
# Copyright (c) 2019 nzanepro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

usage()
{
  echo >&2 "git-archive-submodules - A script to produce an archive tar.gz file of the
                       a git module including all git submodules"
  echo >&2 "requires: git, sed, gzip, and tar (or gtar on macos)"
  echo >&2 "usage: $0 [destination] [recursive]"
}

# requires gnu-tar on mac
export TARCOMMAND=tar
case "$OSTYPE" in
  darwin*)    export TARCOMMAND=gtar;;
  linux-gnu*) export TARCOMMAND=tar;;
  *)        echo "unknown: $OSTYPE" && exit 1;;
esac
command -v ${TARCOMMAND} >/dev/null 2>&1 || { usage; echo >&2 "ERROR: I require ${TARCOMMAND} but it's not installed.  Aborting."; exit 1; }

# reqiures git
command -v git >/dev/null 2>&1 || { usage; echo >&2 "ERROR:I require git but it's not installed.  Aborting."; usage; exit 1; }

# requires sed
command -v sed >/dev/null 2>&1 || { usage; echo >&2 "ERROR:I require sed but it's not installed.  Aborting."; usage; exit 1; }

# requires gzip
command -v gzip >/dev/null 2>&1 || { usage; echo >&2 "ERROR:I require gzip but it's not installed.  Aborting."; usage; exit 1; }

export GIT_TAG=`git describe --tags --abbrev=0`
if [[ -z "${GIT_TAG}" ]]; then
  export GIT_TAG=`git rev-parse --short HEAD`
fi

export TARMODULE=`basename \`git rev-parse --show-toplevel\``
export TARVERSION=`echo ${GIT_TAG} | sed 's/v//g'`
export TARPREFIX="${TARMODULE}-${TARVERSION}"

if [[ ! -d "${TMPDIR}" ]]; then
  export TMPDIR=$(dirname $(mktemp -u))/
fi

# create module archive
git archive --prefix=${TARPREFIX}/ -o ${TMPDIR}/${TARPREFIX}.tar ${GIT_TAG}
if [[ ! -f "${TMPDIR}/${TARPREFIX}.tar" ]]; then
  echo "ERROR: base sourcecode archive was not created. check git output in log above."
  usage
  exit 1
fi

if [[ -z "$2" ]]; then
  recursively=""
else
  recursively=$2
fi

# force init submodules
git submodule update --init ${recursively}

# tar each submodule recursively
git submodule foreach ${recursively} 'git archive --prefix=${TARPREFIX}/${displaypath}/ HEAD > ${TMPDIR}/tmp.tar && ${TARCOMMAND} --concatenate --file=${TMPDIR}/${TARPREFIX}.tar ${TMPDIR}/tmp.tar'

# compress tar file
gzip -9 ${TMPDIR}/${TARPREFIX}.tar
if [[ ! -f "${TMPDIR}/${TARPREFIX}.tar.gz" ]]; then
  echo "ERROR: gzipped archive was not created. check git output in log above."
  usage
  exit 1
fi

# copy file to final name and location if specified
if [[ -z "$1" ]]; then
  destination=${TARPREFIX}.tar.gz
else
  destination=$1
fi
cp ${TMPDIR}/${TARPREFIX}.tar.gz ${destination}
if [[ -f "${TMPDIR}/${TARPREFIX}.tar.gz" ]]; then
  rm ${TMPDIR}/${TARPREFIX}.tar.gz
  echo "created ${destination}"
else
  echo "ERROR copying ${TMPDIR}/${TARPREFIX}.tar.gz to ${destination}"
  usage
  exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
python ${SCRIPT_DIR}/tardotgz-to-zip.py ${destination}
