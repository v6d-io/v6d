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
package io.v6d.hive.ql.io;

import org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hive.conf.HiveConf;
import static org.apache.hadoop.hive.conf.HiveConf.ConfVars.HIVE_ARROW_BATCH_SIZE;

import java.util.Properties;

public class VineyardSerDe extends ArrowColumnarBatchSerDe {
    @Override
    public void initialize(Configuration configuration, Properties tableProperties, Properties partitionProperties)
        throws SerDeException {
        System.out.println("initialize");
        super.initialize(configuration, tableProperties, partitionProperties);
    }

    @Override
    public Object deserialize(Writable writable) {
        return ((RowWritable)writable).getValues();
    }
}