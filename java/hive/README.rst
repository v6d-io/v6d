Hive on Vineyard
================

Environment Setup
-----------------

Using docker to launch the hive server:

.. code:: bash

    docker-compose up -d --force-recreate --remove-orphans

If the result query is large, you may need to increase the memory of the hive server (e.g. Set max memory to 8G):

.. code:: bash

    docker-compose up -d -e SERVICE_OPTS="-Xmx8G" --force-recreate --remove-orphans

Connecting to the hive server:

.. code:: bash

    docker exec -it hive beeline -u 'jdbc:hive2://localhost:10000/;transportMode=http;httpPath=cliservice'

Refer to `apache/hive <https://hub.docker.com/r/apache/hive>`_ for detailed documentation.

Hive Usage
----------

- Create table and insert some data:

    .. code:: sql

        show tables;
        create table hive_example_test(
            a string,
            b int)
        stored as TEXTFILE
        LOCATION "file:///opt/hive/data/warehouse/hive_example_test";

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
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);

- Create table and select

    .. code:: sql

        create table hive_example2(
                    field_1 int,
                    field_2 int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_example2";

        select * from hive_example2;

        explain vectorization only select * from hive_example2;

- Insert using `VineyardSerDe`:

    .. code:: sql

        create table hive_example(
                            field_1 string,
                            field_2 int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);

        select * from hive_example;

- Vectorized Input (and output, currently unavaliabe):

    .. code:: sql

        set hive.fetch.task.conversion=none;
        set hive.vectorized.use.vectorized.input.format=true;
        set hive.vectorized.use.row.serde.deserialize=false;
        set hive.vectorized.use.vector.serde.deserialize=true;
        set hive.vectorized.execution.enabled=true;
        set hive.vectorized.execution.reduce.enabled=true;
        set hive.vectorized.row.serde.inputformat.excludes=io.v6d.hive.ql.io.VineyardInputFormat;

        create table hive_example(
                            field_1 int,
                            field_2 bigint,
                            field_3 boolean,
                            field_4 string,
                            field_5 double,
                            field_6 float)
        row format serde "io.v6d.hive.ql.io.VineyardVectorizedSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardVectorizedInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_example";
        insert into hive_example values(1, 1, true, 'a', 1.0, 1.0), (2, 2, true, 'b', 2.0, 2.0), (3, 3, false, 'c', 3.0, 3.0);

        select * from hive_example1;
        explain vectorization select * from hive_example;

        insert into hive_example values(1, 1), (2, 2), (3,3);

- Test large data sets:

    The following sql statement reads the livejournal dataset (a 27 million line csv file) and stores it in vineyard.
    The dataset must be placed in the correct directory.

    .. code:: sql

        create table hive_example(
                            src_id int,
                            dst_id int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
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

- Test static partition:

    .. code:: sql

        create table hive_static_partition(
            src_id int,
            dst_id int
        )
        partitioned by (value int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat';
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_static_partition";
        insert into table hive_static_partition partition(value=666) values (999, 2), (999, 2), (999, 2);
        insert into table hive_static_partition partition(value=666) values (3, 4);
        insert into table hive_static_partition partition(value=114514) values (1, 2);
        select * from hive_static_partition;
        select * from hive_static_partition where value=666;
        select * from hive_static_partition where value=114514;

- Test dynamic partition:

    .. code:: sql

        create table hive_dynamic_partition_data
        (src_id int,
         dst_id int,
         year int);
        insert into table hive_dynamic_partition_data values (1, 2, 2018),(3, 4, 2018),(1, 2, 2017);

        create table hive_dynamic_partition_test
        (
            src_id int,
            dst_id int
        )partitioned by(mounth int, year int)
        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        stored as
            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat'
        LOCATION "vineyard:///opt/hive/data/warehouse/hive_dynamic_partition_test";
        insert into table hive_dynamic_partition_test partition(mounth=1, year) select src_id,dst_id,year from hive_dynamic_partition_data;
        select * from hive_dynamic_partition_test;

Connect to Hive from Spark
--------------------------

- Download hive 3.1.3, and unpack to somewhere:

  .. code:: bash

      wget https://downloads.apache.org/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz

- Configure Spark session:

  .. code:: scala

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
      val conf = new SparkConf()
      conf.setAppName("Spark on Vineyard")
          // use local executor for development & testing
          .setMaster("local[*]")
          // ensure all executor ready
          .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")

      val spark = SparkSession
          .builder()
          .config(conf)
          .config("hive.metastore.uris", "thrift://localhost:9083")
          .config("hive.metastore.sasl.enabled", "false")
          .config("hive.server2.authentication", "NOSASL")
          .config("hive.metastore.execute.setugi", "false")
          .config("spark.sql.hive.metastore.version", "3.1.3")
          .config("spark.sql.hive.metastore.jars", "path")
          .config("spark.sql.hive.metastore.jars.path", "/opt/apache-hive-3.1.3-bin/lib/*")
          .enableHiveSupport()
          .getOrCreate()
        spark.sql()
      val sc: SparkContext = spark.sparkContext

- Use the session:

  .. code:: scala

      spark.sql(".....")

      sc.stop()

  Refer to `Spark/Hive <https://spark.apache.org/docs/latest/sql-data-sources-hive-tables.html>`_ for detailed documentation.
