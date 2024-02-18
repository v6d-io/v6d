Contributing to vineyard
========================

Vineyard is the product of a dedicated team of software engineers and
researchers. We warmly welcome contributions from the open-source community to
enhance and refine this project!

Vineyard is licensed under the `Apache License 2.0`_.

Install development dependencies
--------------------------------

Vineyard requires the following C++ packages for development:

- apache-arrow >= 0.17.1
- gflags
- glog
- boost
- protobuf
- grpc

and the following python packages that can be easily installed using `pip`:

- libclang
- parsec

Developing Vineyard Using Docker
--------------------------------

To streamline the dependency installation process, we offer a pre-built Docker
image containing all necessary requirements. You can find this image at
`vineyardcloudnative/vineyard-dev <https://hub.docker.com/r/vineyardcloudnative/vineyard-dev/tags>`_.

.. code:: shell

    docker pull vineyardcloudnative/vineyard-dev:latest

Build the source
----------------

You can do an out-of-source build using CMake:

.. code:: shell

    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

The vineyardd target will be generated under the `bin` directory.

You may find building and installation instructions for other platforms from our CI:

- `Ubuntu <https://github.com/v6d-io/v6d/blob/main/.github/workflows/build-compatibility.yml>`_
- `MacOS <https://github.com/v6d-io/v6d/blob/main/.github/workflows/build-compatibility.yml>`_
- `CentOS <https://github.com/v6d-io/v6d/blob/main/.github/workflows/build-centos-latest.yaml>`_
- `Arch Linux <https://github.com/v6d-io/v6d/blob/main/.github/workflows/build-archlinux-latest.yml>`_

Running Unit Tests
------------------

Vineyard incorporates a comprehensive set of unit tests within its continuous integration
process. To build these test cases, execute the following command:

.. code:: shell

    cd build
    make vineyard_tests -j$(nproc)

Before running the test cases, ensure that etcd is properly installed by executing
``brew install etcd`` on macOS or ``pip3 install etcd_distro`` on Linux distributions.

A dedicated script is provided to set up the required environments and execute the test cases:

.. code:: shell

    ./test/runner.py --help
    usage: runner.py [-h] [--with-cpp] [--with-python] [--with-io] [--with-deployment] [--with-migration] [--with-contrib] [--tests [TESTS [TESTS ...]]]

    optional arguments:
      -h, --help            show this help message and exit
      --with-cpp            Whether to run C++ tests
      --with-python         Whether to run python tests
      --with-io             Whether to run IO adaptors tests
      --with-deployment     Whether to run deployment and scaling in/out tests
      --with-migration      Whether to run object migration tests
      --with-contrib        Whether to run python contrib tests
      --tests [TESTS [TESTS ...]]
                            Specify tests cases ro run

As shown above, you could run C++ unittests by

.. code:: shell

    ./test/runner --with-cpp

You could only run specified test case as well:

.. code:: shell

    ./test/runner --with-cpp --tests array_test dataframe_test

Documentation
-------------

Vineyard's documentation is generated using Doxygen and Sphinx. To build the
documentation locally, navigate to the :code:`docs/` directory and execute the
following commands:

.. code:: bash

    cd docs/
    make html

Upon successful completion, the HTML documentation will be available under the
:code:`docs/_build/html` directory:

.. code:: bash

    open _build/html/index.html

For the most up-to-date version of the documentation, visit https://v6d.io.

Vineyard offers comprehensive documentation that delves into the design and
implementation details of the project. The documentation adheres to the syntax
conventions of Doxygen and Sphinx markup. If you identify areas for improvement
or wish to contribute, feel free to submit a pull request. We appreciate your
enthusiasm and support!

Reporting Bugs
--------------

Vineyard is hosted on GitHub and utilizes GitHub issues as its bug tracker.
If you encounter any issues or unexpected behavior while using Vineyard, please `file an issue`_.

Before creating a new bug report, we recommend that you first `search`_ among existing
Vineyard bugs to check if the issue has already been addressed.

When submitting a new bug report, kindly provide essential information regarding your
problem in the description, such as the operating system version, Vineyard version,
and any relevant system configurations. This will greatly assist us in diagnosing
and resolving the issue.

Submitting Pull Requests
------------------------

We greatly appreciate contributions from the community, including bug fixes and new
features. To submit a pull request to Vineyard, please follow the guidelines in this
section:

Install Pre-commit
^^^^^^^^^^^^^^^^^^

Vineyard uses `pre-commit`_ to prevent accidental inclusion of secrets in the Git
repository. To install `pre-commit`_, run:

.. code:: bash

    pip3 install pre-commit

Next, configure the necessary pre-commit hooks with:

.. code:: bash

    pre-commit install

Sign Off Your Commits
^^^^^^^^^^^^^^^^^^^^^

Vineyard has enabled the `DCO`_, which requires you to `sign-off`_ your commits included
in pull requests. Git provides a :code:`-s` command line option to `sign-off`_ your
commit automatically:

.. code:: shell

    git commit -s -m 'This is my commit message'

Code Formatting
^^^^^^^^^^^^^^^

Vineyard adheres to the `Google C++ Style Guide`_. When submitting patches, please format
your code using clang-format with the Makefile command `make vineyard_clformat`, and
ensure your code complies with the cpplint convention using the CMakefile command
`make vineyard_cpplint`.

Open a Pull Request
^^^^^^^^^^^^^^^^^^^

When opening issues or submitting pull requests, please prefix the pull request title
with the issue number and the type of patch (`BUGFIX` or `FEATURE`) in brackets. For
example, `[BUGFIX-1234] Fix crash in sealing vector to vineyard` or ``[FEATURE-2345]
Support seamless operability with PyTorch's tensors``.

Git Workflow for Newcomers
^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally, you do NOT need to rebase your pull requests unless there are merge conflicts
with the main branch. If GitHub indicates "Can’t automatically merge" on your pull
request, you will be asked to rebase your pull request on top of the latest main branch
using the following commands:

+ First, rebase to the most recent main:

  .. code:: shell

      git remote add upstream https://github.com/v6d-io/v6d.git
      git fetch upstream
      git rebase upstream/main

+ If Git shows conflicts, such as in `conflict.cpp`,you need to:
  - Manually modify the file to resolve the conflicts
  - After resolving, mark it as resolved by

  .. code:: shell

      git add conflict.cpp

+ Then, continue rebasing with:

  .. code:: shell

      git rebase --continue

+ Finally, push to your fork, and the pull request will be updated:

  .. code:: shell

      git push --force

Creating a Release
------------------

The Vineyard Python package is built using the `manylinux2014`_ environment. To create
a release version, we utilize Docker for a consistent and reliable build process.
The base image's details can be found in the `docker/pypa/Dockerfile.manylinux1`_ file.

.. _pre-commit: https://pre-commit.com/
.. _file an issue: https://github.com/v6d-io/v6d/issues/new/new
.. _manylinux2014: https://github.com/pypa/manylinux
.. _search: https://github.com/v6d-io/v6d/pulls
.. _CLA: https://cla-assistant.io/v6d-io/v6d
.. _DCO: https://github.com/apps/dco
.. _sign-off: https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s
.. _Google C++ Style Guide: https://google.github.io/styleguide/cppguide.html
.. _docker/pypa/Dockerfile.manylinux1: https://github.com/v6d-io/v6d/blob/main/docker/pypa/Dockerfile.manylinux1
.. _Apache License 2.0: https://github.com/v6d-io/v6d/blob/main/LICENSE
