/** Copyright 2020-2021 Alibaba Group Holding Limited.
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
package io.v6d.modules.basic.arrow;

import static io.v6d.modules.basic.arrow.Arrow.logger;

import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import java.util.ArrayList;
import java.util.List;
import lombok.val;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Hello world! */
public class RecordBatch extends VectorSchemaRoot {
    public static void instantiate() {
        Schema.instantiate();
        Int32Array.instantiate();
        Int64Array.instantiate();
        FloatArray.instantiate();
        DoubleArray.instantiate();
        ObjectFactory.getFactory().register("vineyard::RecordBatch", new RecordBatchResolver());
    }

    public RecordBatch(Schema schema, List<FieldVector> vectors, int nrow) {
        super(schema, vectors, nrow);
    }
}

class RecordBatchResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        val schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));
        val ncol = meta.getIntValue("column_num_");
        val nrow = meta.getIntValue("row_num_");
        logger.debug("batch: ncol = {}, nrow = {}", ncol, nrow);

        logger.debug("meta = {}", meta);

        val vectors = new ArrayList<FieldVector>();
        for (int index = 0; index < meta.getIntValue("__columns_-size"); ++index) {
            val column = meta.getMemberMeta("__columns_-" + index);
            val member = (Array) ObjectFactory.getFactory().resolve(column);
            vectors.add(member.getArray());
        }
        return new RecordBatch(schema, vectors, nrow);
    }
}
