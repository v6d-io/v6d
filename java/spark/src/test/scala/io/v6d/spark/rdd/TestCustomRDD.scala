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

import org.apache.spark.{Partition, SparkConf, SparkContext, TaskContext}
import io.v6d.spark.rdd.RecordFunction._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// Derived from
object TestCustomRDD {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf
      .setAppName("Spark on Vineyard: Custom RDD")
      .setMaster("local[*]")
      // ensure all executor ready
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext

    testCustomRDD(spark, sc)

    sc.stop()
  }

  /** The @spark.createDataFrame@ requires the bean class been defined inside
    * the class, but outside the function.
    *
    * Thus `InternalRecord` cannot be defined both outside the class or inside
    * the function.
    *
    * See also: https://stackoverflow.com/a/34447550/5080177
    */
  case class InternalRecord(itemId: String, itemValue: Double)
      extends Serializable

  def testCustomRDD(spark: SparkSession, sc: SparkContext): Unit = {
    import spark.implicits._

    val rdd = sc.parallelize(1 to 100, 10)
    val recordRDD = rdd.map(item => {
      new Record(item.toString, item.toDouble)
    })
    println("total: ", recordRDD.total)
    val shiftedRDD = recordRDD.shift(10.0)
    println("shifted total: ", shiftedRDD.total)

    val internalRecordRDD =
      recordRDD.map(item => InternalRecord(item.itemId, item.itemValue))

    // custom RDD to DataFrame
    val df = spark.createDataFrame(internalRecordRDD)
    println("df.schema = ", df.schema)
    df.show()

    // SQL on custom RDD
    df.createOrReplaceGlobalTempView("records")
    val result =
      spark.sql("select * from global_temp.records where itemValue < 10")
    result.show()
  }
}

class Record(val itemId: String, val itemValue: Double)
    extends Comparable[Record]
    with Serializable {

  override def compareTo(o: Record): Int = {
    this.itemId.compareTo(o.itemId)
  }

  override def toString: String = {
    s"Record($itemId, $itemValue)"
  }
}

/** Custom RDD.
  */
class RecordRDD(rdd: RDD[Record], arg: Double) extends RDD[Record](rdd) {
  override def compute(
      split: Partition,
      context: TaskContext
  ): Iterator[Record] = {
    firstParent[Record]
      .iterator(split, context)
      .map(record => {
        new Record(record.itemId, record.itemValue + arg)
      })
  }

  override protected def getPartitions: Array[Partition] =
    firstParent[Record].partitions
}

/** Custom functions on custom RDDs
  */
class RecordFunction(rdd: RDD[Record]) {
  def total: Double = rdd.map(_.itemValue).reduce(_ + _)

  def shift(shiftValue: Double): RecordRDD = new RecordRDD(rdd, shiftValue)
}

object RecordFunction {
  implicit def toRecordFunction(rdd: RDD[Record]): RecordFunction =
    new RecordFunction(rdd)
}
