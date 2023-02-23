.. image:: https://v6d.io/_static/vineyard_logo.png
   :target: https://v6d.io
   :align: center
   :alt: vineyard
   :width: 397px

vineyard-io: IO drivers for `vineyard <https://v6d.io>`_
--------------------------------------------------------

vineyard-io is a collection of IO drivers for `vineyard <https://v6d.io>`_. Currently it supports

* Local filesystem
* AWS S3
* Aliyun OSS
* Hadoop filesystem

The vineyard-io package leverages the `filesystem-spec <http://filesystem-spec.readthedocs.io/>`_
to support other storage sinks and sources in a unified fashion. Other adaptors that works for fsspec
could be plugged in as well.
