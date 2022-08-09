/** Copyright 2020-2022 Alibaba Group Holding Limited.
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

import com.google.common.base.Objects;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.val;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Hello world! */
public class Table extends Object {
    private final int rows;
    private final int columns;
    private final Schema schema;
    private final List<RecordBatch> batches;

    public static void instantiate() {
        RecordBatch.instantiate();
        ObjectFactory.getFactory().register("vineyard::Table", new TableResolver());
    }

    public Table(
            final ObjectMeta meta,
            Schema schema,
            int rows,
            int columns,
            List<RecordBatch> batches) {
        super(meta);
        this.schema = schema;
        this.rows = rows;
        this.columns = columns;
        this.batches = batches;
    }

    public Schema getSchema() {
        return schema;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public List<RecordBatch> getBatches() {
        return batches;
    }

    public RecordBatch getBatch(int index) {
        return batches.get(index);
    }

    public VectorSchemaRoot getArrowBatch(int index) {
        return batches.get(index).getBatch();
    }

    @Override
    public boolean equals(java.lang.Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Table that = (Table) o;
        if (this.batches.size() != that.batches.size()) {
            return false;
        }
        for (int index = 0; index < this.batches.size(); ++index) {
            if (!Objects.equal(this.batches.get(index), that.batches.get(index))) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(batches);
    }
}

class TableResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        val ncol = meta.getIntValue("num_columns_");
        val nrow = meta.getIntValue("num_rows_");
        val nbatch = meta.getIntValue("batch_num_");

        val schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));

        val batches =
                IntStream.range(0, nbatch)
                        .mapToObj(
                                index -> {
                                    val batch = meta.getMemberMeta("__batches_-" + index);
                                    return (RecordBatch) ObjectFactory.getFactory().resolve(batch);
                                })
                        .collect(Collectors.toList());
        return new Table(meta, schema, nrow, ncol, batches);
    }
}
