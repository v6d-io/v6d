Deploying using Docker
======================

.. _deploying-using-docker:

Vineyard distributes a docker image that helps the deployment
on platforms where Docker is available. The docker image is hosted
on the Github Packages and can be pulled from:

.. code:: shell

    docker pull vineyardcloudnative/vineyardd:latest

The docker images can be used in the following way

.. code:: shell

    docker run --rm -it vineyardcloudnative/vineyardd:latest

Just like what you can do with a locally installed vineyard package.
See also `Deploying on Linux/MacOS <https://v6d.io/v6d/notes/deploy-docker.html>`_.

Docker images history
---------------------

All history versions can be found from the `ghcr.io/v6d-io/v6d/vineyardd <https://github.com/v6d-io/v6d/pkgs/container/v6d%2Fvineyardd>`_.
