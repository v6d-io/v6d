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
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.util.ObjectResolver;
import io.v6d.modules.basic.columnar.ColumnarData;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.val;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RecordBatch extends Object {
    private static final Logger logger = LoggerFactory.getLogger(RecordBatch.class);

    private final VectorSchemaRoot batch;
    private ObjectResolver resolver;

    public static void instantiate() {
        Schema.instantiate();
        BooleanArray.instantiate();
        Int8Array.instantiate();
        Int16Array.instantiate();
        Int32Array.instantiate();
        Int64Array.instantiate();
        FloatArray.instantiate();
        DoubleArray.instantiate();
        StringArray.instantiate();
        VarBinaryArray.instantiate();
        LargeStringArray.instantiate();
        NullArray.instantiate();
        DateArray.instantiate();
        TimestampArray.instantiate();
        DecimalArray.instantiate();
        ListArray.instantiate();
        StructArray.instantiate();
        MapArray.instantiate();
        ObjectFactory.getFactory().register("vineyard::RecordBatch", new RecordBatchResolver());
    }

    public RecordBatch(final ObjectMeta meta, Schema schema, List<FieldVector> vectors, int nrow) {
        super(meta);
        this.batch = new VectorSchemaRoot(schema.getSchema(), vectors, nrow);
        resolver = new ObjectResolver();
    }

    public VectorSchemaRoot getBatch() {
        return batch;
    }

    public long getRowCount() {
        return batch.getRowCount();
    }

    public long getColumnCount() {
        return batch.getFieldVectors().size();
    }

    public void setResolver(ObjectResolver resolver) {
        this.resolver = resolver;
    }

    public ColumnarData[] columar() {
        List<FieldVector> vectors = batch.getFieldVectors();
        ColumnarData[] columnarData = new ColumnarData[vectors.size()];
        for (int i = 0; i < vectors.size(); i++) {
            columnarData[i] = new ColumnarData(vectors.get(i), resolver);
        }
        return columnarData;
    }

    @Override
    public boolean equals(java.lang.Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        RecordBatch that = (RecordBatch) o;
        return Objects.equal(batch, that.batch);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(batch);
    }
}

class RecordBatchResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        val schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));
        val ncol = meta.getIntValue("column_num_");
        val nrow = meta.getIntValue("row_num_");

        val vectors =
                IntStream.range(0, meta.getIntValue("__columns_-size"))
                        .mapToObj(
                                index -> {
                                    val column = meta.getMemberMeta("__columns_-" + index);
                                    return ((Array) ObjectFactory.getFactory().resolve(column))
                                            .getArray();
                                })
                        .collect(Collectors.toList());
        return new RecordBatch(meta, schema, vectors, nrow);
    }
}
