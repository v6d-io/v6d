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
import io.v6d.core.client.ds.{ObjectFactory, ObjectMeta}
import io.v6d.core.common.util.ObjectID
import io.v6d.modules.basic.arrow.Arrow
import io.v6d.modules.basic.dataframe.{DataFrame => VineyardDataFrame}
import org.apache.spark.sql.vineyard.DataContext

import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.rdd.RDD
import org.apache.spark.{OneToOneDependency, Partition, TaskContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.vectorized.{
  ArrowColumnVector,
  ColumnVector,
  ColumnarBatch
}
import org.apache.spark.sql.internal.SQLConf

import scala.collection.JavaConverters._

/** Provides a RDD to process each partition of Vineyard::GlobalDataFrame.
  *
  * This is useful to convert a Vineyard::GlobalDataFrame to spark.sql.dataframe
  * with mapPartitions.
  */
class GlobalDataFrameChunkRDD(rdd: VineyardRDD)
    extends RDD[VectorSchemaRoot](
      rdd.sparkContext,
      Seq(new OneToOneDependency(rdd))
    ) {
  override def compute(
      split: Partition,
      context: TaskContext
  ): Iterator[VectorSchemaRoot] = {
    // Initialize vineyard context.
    Arrow.instantiate()
    VineyardDataFrame.instantiate()

    val partition = split.asInstanceOf[VineyardPartition]
    firstParent[ObjectID]
      .iterator(split, context)
      .map(record => {
        val meta = partition.client.getMetaData(record)
        val df =
          ObjectFactory.getFactory.resolve(meta).asInstanceOf[VineyardDataFrame]
        df.asBatch
      })
  }

  override protected def getPartitions: Array[Partition] =
    firstParent[VineyardRDD].partitions

  override protected def getPreferredLocations(
      split: Partition
  ): Seq[String] = {
    val partition = split.asInstanceOf[VineyardPartition]
    Seq(partition.host)
  }
}

object GlobalDataFrameChunkRDD {
  def fromVineyardRDD(rdd: VineyardRDD): GlobalDataFrameChunkRDD =
    new GlobalDataFrameChunkRDD(rdd)
}

/** Provides a RDD to process each row of Vineyard::GlobalDataFrame.
  *
  * This is useful to apply RDD-level transformation directly on
  * vineyard::GlobalDataFrame without creating a new spark.sql.DataFrame.
  */
class GlobalDataFrameRDD(rdd: VineyardRDD)
    extends RDD[InternalRow](
      rdd.sparkContext,
      Seq(new OneToOneDependency(rdd))
    ) {

  override def compute(
      split: Partition,
      context: TaskContext
  ): Iterator[InternalRow] = {
    // Initialize vineyard context.
    Arrow.instantiate()
    VineyardDataFrame.instantiate()

    val partition = split.asInstanceOf[VineyardPartition]
    val meta = partition.client.getMetaData(partition.chunkId)
    val batch =
      ObjectFactory.getFactory
        .resolve(meta)
        .asInstanceOf[VineyardDataFrame]
        .asBatch
    val columns: Array[ColumnVector] =
      batch.getFieldVectors.asScala.map(new ArrowColumnVector(_)).toArray
    val columnarBatch = new ColumnarBatch(columns, batch.getRowCount)
    columnarBatch.rowIterator().asScala
  }

  override protected def getPartitions: Array[Partition] =
    firstParent[VineyardRDD].partitions

  override protected def getPreferredLocations(
      split: Partition
  ): Seq[String] = {
    val partition = split.asInstanceOf[VineyardPartition]
    Seq(partition.host)
  }

  def toDF(spark: SparkSession): DataFrame = {
    val globalDataFrameChunkRDD = GlobalDataFrameChunkRDD.fromVineyardRDD(rdd)
    val types = globalDataFrameChunkRDD
      .map(chunk => DataContext.fromArrowSchema(chunk.getSchema))
      .first()
    DataContext.createDataFrame(spark, this, types)
  }
}

object GlobalDataFrameRDD {
  def fromVineyard(rdd: VineyardRDD): GlobalDataFrameRDD =
    new GlobalDataFrameRDD(rdd)

  def makeDataFrame(
      client: IPCClient,
      spark: SparkSession,
      meta: ObjectMeta
  ): DataFrame = {
    val vineyardRDD =
      new VineyardRDD(
        spark.sparkContext,
        meta,
        "partitions_",
        client.getIPCSocket(),
        client.getClusterStatus
      )
    this.fromVineyard(vineyardRDD).toDF(spark)
  }
}
