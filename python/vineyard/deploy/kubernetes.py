#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
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
#

import logging
import os
import re
import tempfile
import time

try:
    import kubernetes
except ImportError:
    kubernetes = None

from .etcd import start_etcd_k8s
from .utils import ensure_kubernetes_namespace

logger = logging.getLogger('vineyard')


def start_vineyardd(
    namespace='vineyard',
    size='512Mi',
    socket='/var/run/vineyard.sock',
    rpc_socket_port=9600,
    vineyard_image='vineyardcloudnative/vineyardd:latest',
    vineyard_image_pull_policy='IfNotPresent',
    vineyard_image_pull_secrets=None,
    k8s_client=None,
):
    """Launch a vineyard cluster on kubernetes.

    Parameters:
        namespace: str
            namespace in kubernetes
        size: int
            The memory size limit for vineyard's shared memory. The memory size
            can be a plain integer or as a fixed-point number using one of these
            suffixes:

            .. code::

                E, P, T, G, M, K.

            You can also use the power-of-two equivalents: Ei, Pi, Ti, Gi, Mi, Ki.

            For example, the following represent roughly the same value:

            .. code::

                128974848, 129k, 129M, 123Mi, 1G, 10Gi, ...
        socket: str
            The UNIX domain socket socket path that vineyard server will listen on.
        rpc_socket_port: int
            The port that vineyard will use to privode RPC service.
        k8s_client: kubernetes.client.api.ApiClient
            A kubernetes client. If not specified, vineyard will try to resolve the
            kubernetes configuration from current context.
        vineyard_image: str
            The docker image of vineyardd to launch the daemonset.
        vineyard_image_pull_policy: str
            The docker image pull policy of vineyardd.
        vineyard_image_pull_secrets: str
            The docker image pull secrets of vineyardd.

    Returns:
        A list of created kubernetes resources during the deploying process. The
        resources can be later release using :meth:`delete_kubernetes_objects`.

    See Also:
        vineyard.deploy.kubernetes.delete_kubernetes_objects
    """
    if kubernetes is None:
        raise RuntimeError('Please install the package python "kubernetes" first')

    if k8s_client is None:
        kubernetes.config.load_kube_config()
        k8s_client = kubernetes.client.ApiClient()

    created_objects = []
    ensure_kubernetes_namespace(namespace, k8s_client=k8s_client)
    created_objects.extend(start_etcd_k8s(namespace, k8s_client=k8s_client))

    with open(
        os.path.join(os.path.dirname(__file__), "vineyard.yaml.tpl"),
        'r',
        encoding='utf-8',
    ) as fp:
        formatter = {
            'Namespace': namespace,
            'Size': size,
            'Socket': socket,
            'Port': rpc_socket_port,
            'Image': vineyard_image,
            'ImagePullPolicy': vineyard_image_pull_policy,
            'ImagePullSecrets': vineyard_image_pull_secrets
            if vineyard_image_pull_secrets
            else 'none',
        }
        definitions = fp.read().format(**formatter)

    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', delete=False
    ) as rendered:
        rendered.write(definitions)
        rendered.flush()
        rendered.close()

        with open(rendered.name, 'r', encoding='utf-8') as fp:
            logger.debug(fp.read())

        created_objects.extend(
            kubernetes.utils.create_from_yaml(
                k8s_client, rendered.name, namespace=namespace
            )
        )
        return created_objects


def recursive_flatten(targets):
    """Flatten the given maybe nested list as a 1-level list."""

    def _recursive_flatten_impl(destination, targets):
        if isinstance(targets, (list, tuple)):
            for target in targets:
                _recursive_flatten_impl(destination, target)
        else:
            destination.append(targets)

    destination = []
    _recursive_flatten_impl(destination, targets)
    return destination


def delete_kubernetes_objects(
    targets, k8s_client=None, verbose=False, wait=False, timeout_seconds=60, **kwargs
):
    """Delete the given kubernetes resources.

    Parameters:
        target: List
            List of Kubernetes objects
        k8s_client:
            The kubernetes client. If not specified, vineyard will try to resolve
            the kubernetes configuration from current context.
        verbose: bool
            Whether to print the deletion logs.
        wait: bool
            Whether to wait for the deletion to complete.
        timeout_seconds: int
            The timeout in seconds for waiting for the deletion to complete.

    See Also:
        vineyard.deploy.kubernetes.start_vineyardd
        vineyard.deploy.kubernetes.delete_kubernetes_object
    """
    for target in recursive_flatten(targets):
        delete_kubernetes_object(
            target,
            k8s_client,
            verbose=verbose,
            wait=wait,
            timeout_seconds=timeout_seconds,
            **kwargs
        )


def delete_kubernetes_object(
    target, k8s_client=None, verbose=False, wait=False, timeout_seconds=60, **kwargs
):
    """Delete the given kubernetes resource.

    Parameters:
        target: object
            The Kubernetes objects that will be deleted.
        k8s_client:
            The kubernetes client. If not specified, vineyard will try to resolve
            the kubernetes configuration from current context.
        verbose: bool
            If True, print confirmation from the delete action. Defaults to False.
        wait: bool
            Whether to wait for the deletion to complete. Defaults to False.
        timeout_seconds: int
            The timeout in seconds for waiting for the deletion to complete. Defaults
            to 60.

    Returns:
        Status: Return status for calls kubernetes delete method.

    See Also:
        vineyard.deploy.kubernetes.start_vineyardd
        vineyard.deploy.kubernetes.delete_kubernetes_objects
    """
    if isinstance(target, (list, tuple)):
        return delete_kubernetes_objects(
            target, k8s_client, verbose=verbose, wait=wait, **kwargs
        )

    if kubernetes is None:
        raise RuntimeError('Please install the package python "kubernetes" first')

    if k8s_client is None:
        kubernetes.config.load_kube_config()
        k8s_client = kubernetes.client.ApiClient()

    group, _, version = target.api_version.partition("/")
    if version == "":
        version = group
        group = "core"
    # Take care for the case e.g. api_type is "apiextensions.k8s.io"
    # Only replace the last instance
    group = "".join(group.rsplit(".k8s.io", 1))
    # convert group name from DNS subdomain format to
    # python class name convention
    group = "".join(word.capitalize() for word in group.split("."))
    fcn_to_call = "{0}{1}Api".format(group, version.capitalize())
    k8s_api = getattr(kubernetes.client, fcn_to_call)(
        k8s_client
    )  # pylint: disable=not-callable

    kind = target.kind
    kind = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", kind)
    kind = re.sub("([a-z0-9])([A-Z])", r"\1_\2", kind).lower()

    try:
        # Expect the user to create namespaced objects more often
        kwargs["name"] = target.metadata.name
        if hasattr(k8s_api, "delete_namespaced_{0}".format(kind)):
            # Decide which namespace we are going to put the object in, if any
            kwargs["namespace"] = target.metadata.namespace
            resp = getattr(k8s_api, "delete_namespaced_{0}".format(kind))(**kwargs)
        else:
            kwargs.pop("namespace", None)
            resp = getattr(k8s_api, "delete_{0}".format(kind))(**kwargs)
    except kubernetes.client.rest.ApiException:
        # Object already deleted.
        return None
    else:
        # Waiting for delete
        if wait:
            start_time = time.time()
            if hasattr(k8s_api, "read_namespaced_{0}".format(kind)):
                while True:
                    try:
                        getattr(k8s_api, "read_namespaced_{0}".format(kind))(**kwargs)
                    except kubernetes.client.rest.ApiException as ex:
                        if ex.status != 404:
                            logger.error(
                                "Deleting %s %s failed: %s",
                                kind,
                                target.metadata.name,
                                str(ex),
                            )
                        break
                    else:
                        time.sleep(1)
                        if time.time() - start_time > timeout_seconds:
                            logger.info(
                                "Deleting %s/%s timeout", kind, target.metadata.name
                            )
        if verbose:
            msg = "{0}/{1} deleted.".format(kind, target.metadata.name)
            if hasattr(resp, "status"):
                msg += " status='{0}'".format(str(resp.status))
            logger.info(msg)
        return resp


__all__ = [
    'start_vineyardd',
    'delete_kubernetes_object',
    'delete_kubernetes_objects',
]
