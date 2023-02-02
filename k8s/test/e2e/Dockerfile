# Copyright 2020-2023 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ghcr.io/v6d-io/v6d/vineyard-python-dev:latest

WORKDIR /

# install kubectl and kubernetes package
RUN python -c "import urllib.request; open('/usr/bin/kubectl', 'wb').write(urllib.request.urlopen('https://dl.k8s.io/release/v1.24.0/bin/linux/amd64/kubectl').read())" && \
    chmod +x /usr/bin/kubectl && \
    pip3 install --no-cache-dir kubernetes

ARG PYPI_DEPENDENCIES=""
ENV PYPI_DEPENDENCIES ${PYPI_DEPENDENCIES}

ARG EXTRA_DEPENDENCIES=""
ENV EXTRA_DEPENDENCIES ${EXTRA_DEPENDENCIES}

RUN if [ ! -z "${PYPI_DEPENDENCIES}" ] && [ ! -z "${EXTRA_DEPENDENCIES}" ]; then pip3 install --no-cache-dir ${PYPI_DEPENDENCIES} --extra-index-url ${EXTRA_DEPENDENCIES}; \
    elif [ ! -z "${PYPI_DEPENDENCIES}" ]; then pip3 install --no-cache-dir ${PYPI_DEPENDENCIES}; \
    fi

ARG APP
ENV APP ${APP}

COPY ${APP} /job.py

ENTRYPOINT [ "python3" ]
CMD [ "/job.py" ]
