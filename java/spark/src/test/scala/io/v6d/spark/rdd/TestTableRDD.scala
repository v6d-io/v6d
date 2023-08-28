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
package io.v6d.spark.rdd

import io.v6d.core.client.IPCClient
import io.v6d.core.common.util.{Env, ObjectID}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object TestTableRDD {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf
      .setAppName("Spark on Vineyard")
      .setMaster("local[*]")
      // ensure all executor ready
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext

    val oid = testBuilder(spark, sc)
    testVineyardRDD(spark, sc, oid.toString())

    sc.stop()
  }

  def testBuilder(
      spark: SparkSession,
      sc: SparkContext
  ): ObjectID = {
    import spark.implicits._

    val SOCKET = Env.getEnv(Env.VINEYARD_IPC_SOCKET)
    val client: IPCClient = new IPCClient(SOCKET)

    val rdd = sc.parallelize(Seq((1.3, 1), (2.6, 2), (3.9, 3)), 3)
    val df = spark.createDataFrame(rdd).toDF("value", "count")
    val dataFrameBuilder = new DataFrameBuilder(client, df)
    dataFrameBuilder.seal(client).getId()
  }

  def testVineyardRDD(
      spark: SparkSession,
      sc: SparkContext,
      input: String
  ): Unit = {
    import spark.implicits._

    val SOCKET = Env.getEnv(Env.VINEYARD_IPC_SOCKET)
    val SOURCE = ObjectID.fromString(input)

    val client: IPCClient = new IPCClient(SOCKET)
    val meta = client.getMetaData(SOURCE)
    val vineyardRDD =
      new VineyardRDD(sc, meta, "partitions_", SOCKET, client.getClusterStatus)

    println(
      "chunks inside a table: ",
      vineyardRDD.collect.mkString("[", ", ", "]")
    )

    val tableRDD = TableRDD.fromVineyard(vineyardRDD)
    val df = tableRDD.toDF(spark)

    df.show()
    println("df.schema = ", df.schema)

    df.createGlobalTempView("count")
    val result = spark.sql("select * from global_temp.count limit 5")
    result.show()
  }
}
