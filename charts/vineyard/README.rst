vineyard
========

Vineyard has been integrated with `Helm`_.

Install
-------

Deploy vineyard as a ``DamonSet`` using ``helm``:

.. code:: bash 

    helm repo add https://dl.bintray.com/libvineyard/charts
    helm install vineyard vineyard -n vineyard --create-namespace

.. _Helm: https://helm.sh/
