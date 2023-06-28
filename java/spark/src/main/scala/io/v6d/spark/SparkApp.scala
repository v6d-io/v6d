/** Copyright 2020-2023 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.v6d.spark

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import io.v6d.core.client.IPCClient
import io.v6d.core.common.util.ObjectID
import io.v6d.spark.rdd.DataFrameBuilder

object SparkApp {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf
      .setAppName("Spark on Vineyard")
      // use local executor for development & testing
      .setMaster("local[*]")
      .set("hive.metastore.warehouse.dir", "/opt/hive/data/warehouse")
      // ensure all executor ready
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")

    val spark = SparkSession
      .builder()
      .config(conf)
      .config("hive.metastore.uris", "thrift://localhost:9083")
      .config("hive.metastore.sasl.enabled", "false")
      .config("hive.server2.authentication", "NOSASL")
      .config("hive.metastore.execute.setugi", "false")
      .config("hive.metastore.warehouse.dir", "/opt/hive/data/warehouse")
      .config("spark.sql.warehouse.dir", "/opt/hive/data/warehouse")
      .config("spark.sql.hive.metastore.version", "2.3.9")
      .config("spark.sql.hive.metastore.jars", "path")
      .config(
        "spark.sql.hive.metastore.jars.path",
        "/opt/apache-hive-2.3.9-bin/lib/*"
      )
      .enableHiveSupport()
      .getOrCreate()

    spark.sql("""
        |drop table if exists hive_example;
        |""".stripMargin)

    // spark.sql(
    //   """
    //     |create table hive_dynamic_partition_test
    //     |        (
    //     |            src_id int,
    //     |            dst_id int
    //     |        )partitioned by(mounth int, year int)
    //     |        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
    //     |        stored as
    //     |            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
    //     |            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat';
    //     |""".stripMargin)
    // spark.sql("select * from hive_dynamic_partition_test;")
    spark
      .sql("""
        |create table hive_example(
        |                            field_1 int,
        |                            field_2 bigint,
        |                            field_3 boolean,
        |                            field_4 string,
        |                            field_5 double,
        |                            field_6 float)
        |        row format serde "io.v6d.hive.ql.io.VineyardSerDe"
        |        stored as
        |            INPUTFORMAT 'io.v6d.hive.ql.io.VineyardInputFormat'
        |            OUTPUTFORMAT 'io.v6d.hive.ql.io.VineyardOutputFormat';
        |""".stripMargin)
      .show()
    spark
      .sql("""
        |insert into hive_example values(1, 1, true, 'a', 1.0, 1.0), (2, 2, true, 'b', 2.0, 2.0), (3, 3, false, 'c', 3.0, 3.0);
        |""".stripMargin)
      .show()
    spark
      .sql("""
        |select * from hive_example;
        |""".stripMargin)
      .show()

    val sc: SparkContext = spark.sparkContext
    sc.stop()
  }
}
