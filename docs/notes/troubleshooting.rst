Troubleshooting
===============

You may have problems when operates vineyard, and content in this page
could helps you when error occurs.

1. *Etcd hasn't been configured properly*

    You may see the following error when sending requests to vineyard:

    .. code::

        Etcd error: etcdserver: too many operations in txn request, error code: 3

    That means your etcd hasn't been configured properly and doesn't support
    more than 128 operations inside a transaction. You could check your etcd
    startup parameters and set :code:`--max-txn-ops` to a larger value, for
    example :code:`102400`.

2. *Encounter strange behaviors when working with pyarrow*

    The python SDK of vineyard depends on libarrow-dev. When the module is imported
    along with pyarrow there will be a tons of DLL conflict issues that need to be
    resolved, see https://issues.apache.org/jira/browse/ARROW-10599 for more details.

    That means, we need to keep the consistence of apache-arrow's installation and
    vineyard's installation. More specifically,

    + For users, just install pyarrow and vineyard both using the :code:`pip` package
      mananger should be enough. You could install them by

        .. code::

            pip3 install pyarrow vineyard

    + For developers of vineyard, the localized installed pyarrow should be built
      from scratch, using the system-wide libarrow-dev, that can be achieved by

        .. code::

            pip3 install --no-binary pyarrow pyarrow

    Besides, you may also meet strange crashes or runtime exceptions if you have
    :code:`import` pyarrow before vineyard, you could try to adjust the import order
    by :code:`import vineyard` before :code:`import pyarrow`.
