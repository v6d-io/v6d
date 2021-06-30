Contributing to vineyard
========================

Vineyard has been developed by an active team of software engineers and
researchers. Any contributions from the open-source community to improve this
project are welcome!

Vineyard is licensed under `Apache License 2.0`_.

Install development dependencies
--------------------------------

Vineyard requires the following C++ packages for development:

- apache-arrow >= 0.17.1
- gflags
- glog
- boost
- gtest, for build test suites
- protobuf
- grpc

and the following python packages that can be easily installed using `pip`:

- libclang
- parsec

Build the source
----------------

You can do an out-of-source build using CMake:

.. code:: shell

    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

The vineyardd target will be generated under the `bin` directory.

Documentation
-------------

Documentation is generated using Doxygen and sphinx. Users can build vineyard's
documentation in the :code:`docs/` directory using:

.. code:: bash

    cd docs/
    make html

The HTML documentation will be available under `docs/_build/html`:

.. code:: bash

    open _build/html/index.html

The latest version of online documentation can be found at https://v6d.io.

Vineyard provides comprehensive documents to explain the underlying
design and implementation details. The documentation follows the syntax
of Doxygen and sphinx markup. If you find anything you can help, submit 
pull request to us. Thanks for your enthusiasm!

Bug report and pull requests
----------------------------

Vineyard is hosted on Github, and use Github issues as the bug tracker.
You can `file an issue`_ when you find anything that is expected to work
with vineyard but it doesn't.

Before creating a new bug entry, we recommend you first `search` among existing
vineyard bugs to see if it has already been resolved.

When creating a new bug entry, please provide necessary information of your
problem in the description, such as operating system version, vineyard
version, and other system configurations to help us diagnose the problem.

We also welcome any help on vineyard from the community, including but not
limited to fixing bugs and adding new features. Our project vineyard has enabled
`DCO`_ thus you will be asked to `sign-off`_ your commits that are included in
your pull requests. Git has a :code:`-s` command line option that can `sign-off`_
your commit automatically:

.. code:: shell

    git commit -s -m 'This is my commit message'

Code format
^^^^^^^^^^^

Vineyard follows the `Google C++ Style Guide`_. When submitting patches
to vineyard, please format your code with clang-format by
the Makefile command `make vineyard_clformat`, and make sure your code doesn't
break the cpplint convention using the CMakefile command `make vineyard_cpplint`.

Open a pull request
^^^^^^^^^^^^^^^^^^^

When opening issues or submitting pull requests, we'll ask you to prefix the
pull request title with the issue number and the kind of patch (`BUGFIX` or `FEATURE`)
in brackets, for example, `[BUGFIX-1234] Fix crash in sealing vector to vineyard`
or `[FEATURE-2345] Support seamless operability with PyTorch's tensors`.

Git workflow for newcomers
^^^^^^^^^^^^^^^^^^^^^^^^^^

You generally do NOT need to rebase your pull requests unless there are merge
conflicts with the main. When Github complaining that "Can’t automatically merge"
on your pull request, you'll be asked to rebase your pull request on top of
the latest main branch, using the following commands:

+ First rebasing to the most recent main:

.. code:: shell

    git remote add upstream https://github.com/v6d-io/v6d.git
    git fetch upstream
    git rebase upstream/main

+ Then git may show you some conflicts when it cannot merge, say `conflict.cpp`,
  you need
  - Manually modify the file to resolve the conflicts
  - After resolved, mark it as resolved by

.. code:: shell

    git add conflict.cpp

+ Then you can continue rebasing by

.. code:: shell

    git rebase --continue

+ Finally push to your fork, then the pull request will be got updated:

.. code:: shell

    git push --force

Cut a release
-------------

The vineyard python package is built using the `manylinux1`_ environments. The
release version is built with Docker. The description of the base image can be
found at `docker/pypa/Dockerfile.manylinux1`_.

.. _file an issue: https://github.com/v6d-io/v6d/issues/new/new
.. _manylinux1: https://github.com/pypa/manylinux
.. _search: https://github.com/v6d-io/v6d/pulls
.. _CLA: https://cla-assistant.io/v6d-io/v6d
.. _DCO: https://github.com/apps/dco
.. _sign-off: https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s
.. _Google C++ Style Guide: https://google.github.io/styleguide/cppguide.html
.. _docker/pypa/Dockerfile.manylinux1: https://github.com/v6d-io/v6d/blob/main/docker/pypa/Dockerfile.manylinux1
.. _Apache License 2.0: https://github.com/v6d-io/v6d/blob/main/LICENSE
