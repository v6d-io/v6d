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

import io.v6d.core.client.ds.{ObjectBuilder, ObjectFactory, ObjectMeta}
import io.v6d.core.client.{Client, IPCClient}
import io.v6d.core.common.util.{ObjectID, VineyardException}
import io.v6d.modules.basic.arrow.{Arrow, RecordBatchBuilder, Schema, SchemaBuilder}
import io.v6d.modules.basic.dataframe.{DataFrame => VineyardDataFrame}
import org.apache.spark.rdd.RDD
import org.apache.spark.{OneToOneDependency, Partition, TaskContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.vectorized.{ArrowColumnVector, ColumnVector, ColumnarBatch}
import org.apache.spark.sql.vineyard.DataContext
import org.apache.spark.sql.internal.SQLConf

import scala.collection.JavaConverters._


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
      ObjectFactory.getFactory.resolve(meta).asInstanceOf[VineyardDataFrame].asBatch
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
  def fromVineyard(rdd: VineyardRDD): GlobalDataFrameRDD = new GlobalDataFrameRDD(rdd)
}

class GlobalDataFrameBuilder(
  private val client: IPCClient,
  private val sdf: DataFrame
) extends ObjectBuilder () {

  @throws(classOf[VineyardException])
  override def build(client:Client) = {}

  @throws(classOf[VineyardException])
  override def seal(client:Client): ObjectMeta = {
    this.build(client);
    val meta = ObjectMeta.empty();
    val timeZoneId = SQLConf.get.sessionLocalTimeZone
    val arrowSchema = DataContext.toArrowSchema(sdf.schema, timeZoneId)
    val schemaBuilder = SchemaBuilder.fromSchema(arrowSchema)
    val batches: Array[ObjectID] = sdf.rdd.zipWithIndex.mapPartitions(iterator => {
      val recordBatchBuilder = new RecordBatchBuilder(this.client, arrowSchema, iterator.length);
      recordBatchBuilder.finishSchema(this.client);
      iterator.foreach { case (row, rowId) =>
        row.schema.toList.zipWithIndex.foreach { case (field, fid) =>
          val builder = recordBatchBuilder.getColumnBuilder(fid)
          if (field.dataType.isInstanceOf[BooleanType]) {
            builder.setBoolean(rowId.toInt, row.getBoolean(fid))
          } else if (field.dataType.isInstanceOf[IntegerType]) {
            builder.setInt(rowId.toInt, row.getInt(fid))
          } else if (field.dataType.isInstanceOf[LongType]) {
            builder.setLong(rowId.toInt, row.getLong(fid))
          } else if (field.dataType.isInstanceOf[FloatType]) {
            builder.setFloat(rowId.toInt, row.getFloat(fid))
          } else if (field.dataType.isInstanceOf[DoubleType]) {
            builder.setDouble(rowId.toInt, row.getDouble(fid))
          } else {
            throw new Exception("Columnar builder for type " + field.dataType + " is not supported")
          }
        }
      }
      Iterable(recordBatchBuilder.seal(client).getId).iterator
    }).collect()
    meta.setTypename("vineyard::Table")
    meta.setValue("batch_num_", batches.length)
    meta.setValue("num_rows_", -1) // FIXME
    meta.setValue("num_columns_", sdf.schema.size)
    meta.addMember("schema", schemaBuilder.seal(client))
    meta.setGlobal()
    meta.setValue("__partitions_-size", batches.length);
    for((batch, i) <- batches.zipWithIndex) {
      meta.addMember("__partitions_-" + i, batch)
    }
    meta.setNBytes(0) // FIXME
    return this.client.createMetaData(meta)
  }
}
