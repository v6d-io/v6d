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
      .setMaster("local[*]")
      // ensure all executor ready
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = spark.sparkContext

    sc.stop()
  }
}
