Hive on Vineyard
================

Environment Setup
-----------------

Using docker to launch the hive server:

.. code:: bash

    export HIVE_VERSION=4.0.0-alpha-2
    docker run \
        --rm \
        -it \
        -p 10000:10000 \
        -p 10002:10002 \
        -v `pwd`/hive-warehouse:/opt/hive/data/warehouse \
        -v `pwd`/java/hive/target:/opt/hive/auxlib \
        --env HIVE_AUX_JARS_PATH=/opt/hive/auxlib/ \
        --env SERVICE_NAME=hiveserver2 \
        --name hive \
        apache/hive:${HIVE_VERSION}

Connecting to the hive server:

.. code:: bash

    docker exec -it hive beeline -u 'jdbc:hive2://localhost:10000/'

Refer to `apache/hive <https://hub.docker.com/r/apache/hive>`_ for detailed documentation.

Hive Usage
----------

- Create table and insert some data:

    .. code:: sql

        show tables;
        create table hive_example(
            a string,
            b int)
        stored as TEXTFILE
        LOCATION "file:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);
        select count(distinct a) from hive_example;
        select sum(b) from hive_example;

- Inspect the store file of hive table:

    .. code:: sql

        describe formatted hive_example;

Hive and Vineyard
-----------------

- Create hive table on vineyard:

    .. code:: sql

        create table hive_example(
            a string,
            b int)
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_example";
