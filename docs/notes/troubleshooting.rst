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
