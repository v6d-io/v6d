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
        --env SERVICE_OPTS="-Djnr.ffi.asm.enabled=false" \
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

- Create hive table on vineyard (using :code:`file:///` is enough as we won't touch filesystem input/output format):

    .. code:: sql

        create table hive_example(
            a string,
            b int)
        row format serde "org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "file:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);

- Create table and select

    .. code:: sql

        create table hive_example2(
                    field_1 int,
                    field_2 int)
        row format serde "org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "file:///opt/hive/data/warehouse/hive_example2";

        select * from hive_example;

        explain vectorization only select * from hive_example;

- Insert using `VineyardSerDe`:

    .. code:: sql

        create table hive_example(
                            field_1 int,
                            field_2 int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "file:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);

- Vectorized Input (and output):

    .. code:: sql

        set hive.fetch.task.conversion=none;
        set hive.vectorized.use.vectorized.input.format=true;
        set hive.vectorized.use.row.serde.deserialize=false;
        set hive.vectorized.use.vector.serde.deserialize=true;
        set hive.vectorized.execution.enabled=true;
        set hive.vectorized.execution.reduce.enabled=true;
        set hive.vectorized.row.serde.inputformat.excludes=io.v6d.hive.ql.io.VineyardBatchInputFormat;

        create table hive_example(
                            field_1 int,
                            field_2 int)
        row format serde "org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardBatchInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "file:///opt/hive/data/warehouse/hive_example";

        select * from hive_example;
        explain vectorization select * from hive_example;

        insert into hive_example values('a', 1), ('a', 2), ('b',3);

- Test large data sets:

    Test large data sets must point out the `hive.arrow.batch.size` to avoid etcd failure. The default value is 1000.
    The recommended value for the field is one-tenth of the number of rows in the table.
    The following sql statement reads the livejournal dataset (a 27 million line csv file) and stores it in vineyard.
    You must place the dataset in the correct directory.

    .. code:: sql

        set hive.fetch.task.conversion=none;
        set hive.vectorized.use.vectorized.input.format=true;
        set hive.vectorized.use.row.serde.deserialize=false;
        set hive.vectorized.use.vector.serde.deserialize=true;
        set hive.vectorized.execution.enabled=true;
        set hive.vectorized.execution.reduce.enabled=true;
        set hive.vectorized.row.serde.inputformat.excludes=io.v6d.hive.ql.io.VineyardBatchInputFormat;
        set hive.arrow.batch.size=2000000;

        create table hive_example(
                            src_id int,
                            dst_id int)
        row format serde "org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardBatchInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat';
        create table hive_test_data_livejournal(
                            src_id int,
                            dst_id int
        )
        row format serde 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        stored as textfile ;
        load data local inpath "file:///opt/hive/data/warehouse/soc-livejournal.csv" into table hive_test_data_livejournal;
        insert into hive_example select * from hive_test_data_livejournal; 

- Test output format:

    .. code:: sql

        create table hive_example_orc(
                                    field_1 int,
                                    field_2 int)
        stored as orc
        LOCATION "file:///opt/hive/data/warehouse/hive_example_orc";
        insert into hive_example values(1, 1), (2, 2), (3, 3);
        explain vectorization select * from hive_example_orc;


