.. _divein-driver-label:

I/O Drivers
===========

As we have shown in the getting-started, the ``open`` function in vineyard can open a local
file as a stream  for consuming, and we notice that the path of the local file is headed
with the scheme ``file://``.

Actually, vineyard supports several different types of data source, e.g., ``kafka://``
for kafka topics. The functional methods to open different data sources as vineyard
streams are called ``drivers`` in vineyard. They are registered to ``open`` for
specific schemes, so that when ``open`` is invoked, it will dispatch the corresponding
driver to handle the specific data source according to the scheme of the path.

The following sample code demonstrates the dispatching logic in ``open``, and the
registration examples.

.. code:: python

    >>> @registerize
    >>> def open(path, *args, **kwargs):
    >>>     scheme = urlparse(path).scheme

    >>>     for reader in open._factory[scheme][::-1]:
    >>>         r = reader(path, *args, **kwargs)
    >>>         if r is not None:
    >>>             return r
    >>>     raise RuntimeError('Unable to find a proper IO driver for %s' % path)
    >>>
    >>> # different driver functions are registered as follows
    >>> open.register('file', local_driver)


Most importantly, the registration design allows users to register their own  drivers
to ``registerized`` vineyard methods using ``.register``, which prevents major revisions
on the processing code to fulfill customized computation requirements.
