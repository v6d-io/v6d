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
package org.apache.spark.sql.vineyard

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, Schema}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.{InternalRow, JavaTypeInference}
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.util.ArrowUtils

object DataContext extends Logging {
  def getSchema(
      spark: SparkSession,
      beanClass: Class[_]
  ): Seq[AttributeReference] = {
    val (dataType, _) = JavaTypeInference.inferDataType(beanClass)
    dataType
      .asInstanceOf[StructType]
      .fields
      .map(f => AttributeReference(f.name, f.dataType, f.nullable)())
  }

  def createDataFrame(
      spark: SparkSession,
      rdd: RDD[InternalRow],
      schema: StructType,
      isStreaming: Boolean = false
  ): DataFrame = {
    spark.internalCreateDataFrame(rdd, schema, isStreaming)
  }

  def toArrowType(dt: DataType, timeZoneId: String): ArrowType = {
    ArrowUtils.toArrowType(dt, timeZoneId)
  }

  def fromArrowType(dt: ArrowType): DataType = {
    ArrowUtils.fromArrowType(dt)
  }

  def toArrowField(
      name: String,
      dt: DataType,
      nullable: Boolean,
      timeZoneId: String
  ): Field = {
    ArrowUtils.toArrowField(name, dt, nullable, timeZoneId)
  }

  def fromArrowField(field: Field): DataType = {
    ArrowUtils.fromArrowField(field)
  }

  def toArrowSchema(schema: StructType, timeZoneId: String): Schema = {
    ArrowUtils.toArrowSchema(schema, timeZoneId)
  }

  def fromArrowSchema(schema: Schema): StructType = {
    ArrowUtils.fromArrowSchema(schema)
  }
}
