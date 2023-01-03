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

import io.v6d.core.client.ds.ObjectMeta
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.internal.Logging
import io.v6d.core.client.{ClusterStatus, IPCClient}
import io.v6d.core.common.util.{InstanceID, ObjectID}

import scala.collection.JavaConverters._

class VineyardPartition(
    val index: Int,
    val chunkId: ObjectID,
    val instanceId: InstanceID,
    val host: String,
    val socket: String
) extends Partition
    with Logging {

  @transient lazy val client: IPCClient = {
    if (socket.isEmpty) {
      new IPCClient()
    } else {
      new IPCClient(socket)
    }
  }
}

/** Construct a vineyard RDD using the vineyard client.
  *
  * @param pattern:
  *   pattern for the chunk keys
  */
class VineyardRDD(
    sc: SparkContext,
    private val meta: ObjectMeta,
    private val pattern: String, // pattern for the chunk keys
    private val socket: String,
    private val cluster: ClusterStatus
) extends RDD[ObjectID](sc, Nil) {
  val chunks: Array[ObjectMeta] =
    meta.iteratorMembers(pattern).asScala.map(_.getValue).toArray

  override def compute(
      split: Partition,
      context: TaskContext
  ): Iterator[ObjectID] = {
    val partition = split.asInstanceOf[VineyardPartition]
    Iterator(partition.chunkId)
  }

  override protected def getPartitions: Array[Partition] = {
    val partitions = new Array[Partition](chunks.length)
    for (i <- chunks.indices) {
      partitions(i) = new VineyardPartition(
        i,
        chunks(i).getId,
        meta.getInstanceId,
        cluster.getInstance(meta.getInstanceId).getHostname,
        socket
      )
    }
    partitions
  }

  override protected def getPreferredLocations(
      split: Partition
  ): Seq[String] = {
    val partition = split.asInstanceOf[VineyardPartition]
    Seq(partition.host)
  }
}
