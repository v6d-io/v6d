Hive on Vineyard
================

Environment Setup
-----------------

Using docker to launch the hive server:

.. code:: bash

    docker-compose -f ./docker/docker-compose.yaml up -d --force-recreate --remove-orphans

If the result query is large, you may need to increase the memory of the hive server (e.g. Set max memory to 8G):

.. code:: bash

    docker-compose -f ./docker/docker-compose.yaml up -d -e SERVICE_OPTS="-Xmx8G" --force-recreate --remove-orphans

Connecting to the hive server:

.. code:: bash

    docker exec -it hive beeline -u 'jdbc:hive2://localhost:10000/;transportMode=http;httpPath=cliservice'

Refer to `apache/hive <https://hub.docker.com/r/apache/hive>`_ for detailed documentation.

Hive Usage
----------

In this repo, we set ```hive.default.fileformat``` as ```Vineyard``` and set ```hive.metastore.warehouse.dir``` as 
```vineyard:///user/hive/warehouse```(In java/hive/conf/hive-size.xml), so the default storage format of hive is vineyard.
If you want to use local file system or HDFS, you need to change the configuration or point out the storage format when
creating table.

- Create table as textfile and insert some data:

    .. code:: sql

        show tables;
        create table hive_example(
            a string,
            b int)
        stored as TEXTFILE
        location "file:///opt/hive/data/warehouse/hive_example";

        insert into hive_example values('a', 1), ('a', 2), ('b',3);
        select count(distinct a) from hive_example;
        select sum(b) from hive_example;

- Inspect the store file of hive table:

    .. code:: sql

        describe formatted hive_example;

Hive and Vineyard
-----------------

- Start vineyard server:

    The socket file must be placed in the correct directory. Please refer to the docker-compose.yml file for details.
    You can change the socket file path as you like and change the docker-compose.yml file accordingly.

      .. code:: bash
  
          vineyardd --socket=./vineyard/vineyard.sock --meta=local

- Create hive table on vineyard:

    .. code:: sql

        create table hive_example(
            a string,
            b int);
        describe formatted hive_example;
        drop table hive_example;

- Create table and select

    .. code:: sql

        create table hive_example2(
                    field_1 string,
                    field_2 int);
        insert into hive_example2 values('a', 1), ('b', 2), ('c', 3);
        select * from hive_example2;

        explain vectorization only select * from hive_example2;
        drop table hive_example2;

- Vectorized input and output(Currently unavaliabe):

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
        drop table hive_example;

- Test large data sets:

    The following sql statement reads the livejournal dataset (a 27 million line csv file) and stores it in vineyard.
    The dataset must be placed in the correct directory.

    .. code:: sql

        create table hive_example3(
                            src_id int,
                            dst_id int);
        create table hive_test_data_livejournal(
                            src_id int,
                            dst_id int
        )
        row format serde 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        stored as textfile;
        load data local inpath "file:///opt/hive/data/warehouse/soc-livejournal.csv" into table hive_test_data_livejournal;
        insert into hive_example3 select * from hive_test_data_livejournal;
        drop table hive_test_data_livejournal;
        select * from hive_example3;

- Test static partition:

    .. code:: sql

        create table hive_static_partition(
            src_id int,
            dst_id int
        ) partitioned by (value int);
        insert into table hive_static_partition partition(value=666) values (3, 4);
        insert into table hive_static_partition partition(value=666) values (999, 2), (999, 2), (999, 2);
        insert into table hive_static_partition partition(value=114514) values (1, 2);
        select * from hive_static_partition;
        select * from hive_static_partition where value=666;
        select * from hive_static_partition where value=114514;
        drop table hive_static_partition;

- Test dynamic partition:

    .. code:: sql

        create table hive_dynamic_partition_data(
            src_id int,
            dst_id int,
            year int)
        stored as TEXTFILE
        location "file:///opt/hive/data/warehouse/hive_dynamic_partition_data";
        insert into table hive_dynamic_partition_data values (1, 2, 2018),(3, 4, 2018),(1, 2, 2017);

        create table hive_dynamic_partition_test
        (
            src_id int,
            dst_id int
        )partitioned by(mounth int, year int);
        insert into table hive_dynamic_partition_test partition(mounth=1, year) select src_id,dst_id,year from hive_dynamic_partition_data;
        select * from hive_dynamic_partition_test;
        drop table hive_dynamic_partition_test;
        drop table hive_dynamic_partition_data;

- Test all primitive types:
  
    Now vineyard support to store tinyint, smallint, int, bigint, boolean, string, float, double, date, timestamp, binary and decimal.

    .. code:: sql
  
        create table test_all_primitive_types (
            field_1 tinyint,
            field_2 smallint,
            field_3 bigint,
            field_4 int,
            field_5 double,
            field_6 float,
            field_7 string,
            field_9 varchar(10),
            field_10 char(10),
            field_8 binary,
            field_11 date,
            field_12 boolean,
            field_13 timestamp,
            field_14 decimal(6, 2)
        );

        insert into test_all_primitive_types select
            tinyint(1),
            smallint(1),
            42,
            bigint(1),
            double(2.0),
            float(1.0),
            'hello world1!',
            'hello world2!',
            'hello world3!',
            cast('hello world4!' as binary),
            date('2023-12-31'),
            true,
            timestamp('2023-12-31 23:59:59'),
            cast(1234.56 as decimal);

        select * from test_all_primitive_types;
        drop table test_all_primitive_types;

- Test nested types:

    Now vineyard support to store array, map and struct.

    .. code:: sql

        CREATE TABLE nested_table (
            field_1 map<int,
                        array<struct<field_1:int,
                                     field_2:string>>>
        );

        insert INTO nested_table select
            map(
                42,
                array(named_struct('field_1', 1,
                                   'field_2', 'hello'),
                      named_struct('field_1', 2,
                                   'field_2', 'world!')));

        select * from nested_table;
        drop table nested_table;

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

Build Hive Docker Image with Hadoop
-----------------

### Prepare vineyard jars
```bash
    # Currently, the vineyard jar cannot run directly on hive because of
    # dependency conflicts. You can run it temporarily by reverting to an
    # older version of guava (such as 14.0.1) dependent by vineyard.
    # This problem will be fixed in the future.
    cd v6d/java
    mvn clean package
```

### Build docker images
```bash
    cd v6d/java/hive/docker
    ./build.sh
```

### Create network
```bash
    docker network create hadoop-network
```

### Start sql server for hive metastore
```bash
    cd v6d/java/hive/docker/dependency/mysql
    docker-compose -f mysql-compose.yaml up -d
    # You can change the password in mysql-compose.yaml and hive-site.xml
```

Using vineyard as storage
-----------------

### Run vineyardd
```bash
    cd v6d/build

    # at terminal 1
    ./bin/vineyardd --socket=~/vineyard_sock/0/vineyard.sock -rpc_socket_port=9601 --etcd_endpoint="0.0.0.0:2382"

    # at terminal 2
    ./bin/vineyardd --socket=~/vineyard_sock/1/vineyard.sock -rpc_socket_port=9602 --etcd_endpoint="0.0.0.0:2382"

    # at terminal 3
    ./bin/vineyardd --socket=~/vineyard_sock/2/vineyard.sock -rpc_socket_port=9603 --etcd_endpoint="0.0.0.0:2382"

    # at terminal 4
    ./bin/vineyardd --socket=~/vineyard_sock/metastore/vineyard.sock -rpc_socket_port=9604 --etcd_endpoint="0.0.0.0:2382"

    # at terminal 5
    ./bin/vineyardd --socket=~/vineyard_sock/hiveserver/vineyard.sock -rpc_socket_port=9605 --etcd_endpoint="0.0.0.0:2382"
```

### Copy vineyard jars to share dir
```bash
    mkdir -p v6d/share
    cd v6d/java/hive
    # you can change share dir in docker-compose.yaml
    cp target/vineyard-hive-0.1-SNAPSHOT.jar ../../../share
```

### Run hadoop & hive docker images
```bash
    cd v6d/java/hive/docker
    docker-compose -f docker-compose-distributed.yaml up -d
```

### Create table
```bash
    docker exec -it hive-hiveserver2 beeline -u "jdbc:hive2://hive-hiveserver2:10000" -n root
```

```sql
    -- in beeline
    drop table test_hive;
    create table test_hive(field int);
    insert into table test_hive values (1),(2),(3),(4),(5),(6),(7),(8),(9),(10);
    select * from test_hive;
```
