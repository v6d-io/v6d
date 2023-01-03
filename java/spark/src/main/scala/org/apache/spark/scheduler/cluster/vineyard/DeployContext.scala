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
package org.apache.spark.scheduler.cluster.vineyard

import org.apache.spark.internal.Logging
import org.apache.spark.{SparkConf, SparkContext, TaskContext}
import org.apache.spark.scheduler.cluster.CoarseGrainedSchedulerBackend
import org.apache.spark.scheduler.cluster.ExecutorData

import scala.collection.mutable

object DeployContext extends Logging {

  def getMaxCores(conf: SparkConf): String = {
    val maxCores = conf.getOption("spark.cores.max")
    if (maxCores.isDefined) {
      maxCores.get
    } else {
      conf
        .getOption("spark.deploy.defaultCores")
        .getOrElse(Int.MaxValue.toString)
    }
  }

  def isDriver: Boolean = {
    // sc is not available on worker
    //
    // sc.getConf.getOption("spark.driver.host").isEmpty
    TaskContext.get() == null
  }

  def getExecutors(sc: SparkContext): mutable.HashMap[String, ExecutorData] = {
    if (!sc.schedulerBackend.isInstanceOf[CoarseGrainedSchedulerBackend]) {
      return mutable.HashMap[String, ExecutorData]()
    }

    // scala reflection won't work as it is a private member
    val backend =
      sc.schedulerBackend.asInstanceOf[CoarseGrainedSchedulerBackend]
    val field =
      "org$apache$spark$scheduler$cluster$CoarseGrainedSchedulerBackend$$executorDataMap"
    val accessor = backend.getClass.getSuperclass.getDeclaredField(field)
    accessor.setAccessible(true)
    accessor.get(backend).asInstanceOf[mutable.HashMap[String, ExecutorData]]
  }
}
