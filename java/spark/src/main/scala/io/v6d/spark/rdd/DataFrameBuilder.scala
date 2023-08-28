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

import io.v6d.core.client.{Client, IPCClient}
import io.v6d.core.client.ds.{ObjectBuilder, ObjectFactory, ObjectMeta}
import io.v6d.core.common.util.{ObjectID, VineyardException}
import io.v6d.modules.basic.arrow.{
  Arrow,
  RecordBatchBuilder,
  Schema,
  SchemaBuilder
}
import org.apache.spark.sql.vineyard.DataContext

import org.apache.spark.{OneToOneDependency, Partition, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.vectorized.{
  ArrowColumnVector,
  ColumnVector,
  ColumnarBatch
}
import org.apache.spark.sql.types._
import org.apache.spark.sql.internal.SQLConf
import org.apache.arrow.vector.VectorSchemaRoot

import scala.collection.JavaConverters._

/** Provides a Builder to put spark.sql.DataFrame int vineyard as
  * Vineyard::Table.
  */
class DataFrameBuilder(
    private val client: IPCClient,
    private val sparkDF: DataFrame
) extends ObjectBuilder() {

  @throws(classOf[VineyardException])
  override def build(client: Client): Unit = {}

  // scalastyle:off method.length
  @throws(classOf[VineyardException])
  override def seal(client: Client): ObjectMeta = {
    this.build(client);
    val meta = ObjectMeta.empty();
    val timeZoneId = SQLConf.get.sessionLocalTimeZone
    val schema = sparkDF.schema
    val arrowSchema = DataContext.toArrowSchema(schema, timeZoneId)
    val schemaBuilder = SchemaBuilder.fromSchema(arrowSchema)
    val sock = client.getIPCSocket()
    val batches: Array[ObjectID] = sparkDF.rdd
      .mapPartitions(iterator => {
        val localArray = iterator.toArray
        val localClient = new IPCClient(sock)
        val localArrowSchema = DataContext.toArrowSchema(schema, timeZoneId)
        val recordBatchBuilder = new RecordBatchBuilder(
          localClient,
          localArrowSchema,
          localArray.length
        );
        recordBatchBuilder.finishSchema(localClient);
        localArray.zipWithIndex.foreach { case (row, rowId) =>
          schema.toList.zipWithIndex.foreach { case (field, fid) =>
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
              throw new Exception(
                "Columnar builder for type " + field.dataType + " is not supported"
              )
            }
          }
        }
        val batchMeta = recordBatchBuilder.seal(localClient)
        val id = batchMeta.getId
        Iterable(id).iterator
      })
      .collect()
    meta.setTypename("vineyard::Table")
    meta.setValue("batch_num_", batches.length)
    meta.setValue("num_rows_", -1)
    meta.setValue("num_columns_", sparkDF.schema.size)
    meta.addMember("schema_", schemaBuilder.seal(client))
    meta.setGlobal()
    meta.setValue("partitions_-size", batches.length);
    for ((batch, i) <- batches.zipWithIndex) {
      meta.addMember("partitions_-" + i, batch)
    }
    meta.setNBytes(0)
    client.createMetaData(meta)
  }
  // scalastyle:on method.length
}
