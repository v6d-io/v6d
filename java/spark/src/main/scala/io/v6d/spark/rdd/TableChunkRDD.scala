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

import io.v6d.core.client.ds.ObjectFactory
import io.v6d.core.common.util.ObjectID
import io.v6d.modules.basic.arrow.{Arrow, RecordBatch}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.spark.rdd.RDD
import org.apache.spark.{OneToOneDependency, Partition, TaskContext}

class TableChunkRDD(rdd: VineyardRDD)
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

    val partition = split.asInstanceOf[VineyardPartition]
    firstParent[ObjectID]
      .iterator(split, context)
      .map(record => {
        val meta = partition.client.getMetaData(record)
        val batch =
          ObjectFactory.getFactory.resolve(meta).asInstanceOf[RecordBatch]
        batch.getBatch
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

object TableChunkRDD {
  def fromVineyardRDD(rdd: VineyardRDD): TableChunkRDD = new TableChunkRDD(rdd)
}
