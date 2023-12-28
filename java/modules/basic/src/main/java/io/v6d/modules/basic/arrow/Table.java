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
package io.v6d.modules.basic.arrow;

import com.google.common.base.Objects;
import io.v6d.core.client.Context;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.val;
import org.apache.arrow.vector.VectorSchemaRoot;

public class Table extends Object {
    private final int rows;
    private final int columns;
    private final Schema schema;
    private final List<RecordBatch> batches;
    private List<ObjectMeta> batchMetas;
    private final int batchNum;

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
        this.batchNum = batches.size();
        batchMetas = new java.util.ArrayList<>();
        for (int i = 0; i < this.batchNum; i++) {
            batchMetas.add(meta.getMemberMeta("partitions_-" + i));
        }
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

    public int getBatchNum() {
        return batchNum;
    }

    public synchronized RecordBatch getBatch(int index) {
        if (batches.get(index) == null) {
            // Batch is not at local. Find it from remote.
            ObjectMeta batchMeta;
            try {
                batchMeta =
                        Context.getClient()
                                .getMetaData(
                                        meta.getMemberMeta("partitions_-" + index).getId(), true);
            } catch (VineyardException e) {
                Context.println("Get remote batch failed!");
                return null;
            }
            batches.set(index, (RecordBatch) ObjectFactory.getFactory().resolve(batchMeta));
        }
        return batches.get(index);
    }

    public synchronized String[] getHostsOfRecordBatches(int start, int length) {
        if (start + length > batchNum) {
            return null;
        }
        String[] hosts = new String[length];
        for (int i = 0; i < length; i++) {
            hosts[i] = batchMetas.get(start + i).getStringValue("host_");
        }
        return hosts;
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
        val nbatch = meta.getIntValue("partitions_-size");

        val schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));

        val batches =
                IntStream.range(0, nbatch)
                        .mapToObj(
                                index -> {
                                    val batch = meta.getMemberMeta("partitions_-" + index);
                                    if (batch.getInstanceId().compareTo(Context.getInstanceID())
                                            == 0) {
                                        return (RecordBatch)
                                                ObjectFactory.getFactory().resolve(batch);
                                    }
                                    return null;
                                })
                        .collect(Collectors.toList());
        return new Table(meta, schema, nrow, ncol, batches);
    }
}
