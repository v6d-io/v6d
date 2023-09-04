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
package io.v6d.modules.basic.dataframe;

import com.fasterxml.jackson.databind.JsonNode;
import com.google.common.collect.ImmutableList;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.*;
import org.apache.arrow.util.Collections2;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

public class DataFrame extends Object {
    private int rowCount;
    private int columnCount;
    private List<JsonNode> columns;
    private List<Tensor> values;

    public static void instantiate() {
        Tensor.instantiate();
        ObjectFactory.getFactory().register("vineyard::DataFrame", new DataFrameResolver());
    }

    public DataFrame(
            final ObjectMeta meta,
            int rowCount,
            int columnCount,
            final List<JsonNode> columns,
            final List<Tensor> values) {
        super(meta);
        this.rowCount = rowCount;
        this.columnCount = columnCount;
        this.columns = columns;
        this.values = values;
    }

    public DataFrame() {
        super(ObjectMeta.empty());
        this.rowCount = 0;
        this.columnCount = 0;
        this.columns = new ArrayList<>();
        this.values = new ArrayList<>();
    }

    public int getRowCount() {
        return this.rowCount;
    }

    public int getColumnCount() {
        return this.columnCount;
    }

    public List<Tensor> values() {
        return values;
    }

    public List<JsonNode> columns() {
        return columns;
    }

    public Tensor value(int index) {
        return values.get(index);
    }

    public void setColumns(List<JsonNode> columns) {
        this.columns = columns;
    }

    public void setRowCount(int rowCount) {
        this.rowCount = rowCount;
    }

    public void setColumnCount(int columnCount) {
        this.columnCount = columnCount;
    }

    public void setValues(List<Tensor> values) {
        this.values = values;
    }

    public ValueVector valueArray(int index) {
        return values.get(index).getArray();
    }

    public Schema schema() {
        List<Field> fields = new ArrayList<>();
        for (int index = 0; index < columnCount; ++index) {
            val field = values.get(index).getArray().getField();
            val name = columns.get(index).asText();
            fields.add(
                    new Field(
                            name,
                            field.getFieldType(),
                            Collections2.immutableListCopy(field.getChildren())));
        }
        return new Schema(fields);
    }

    public VectorSchemaRoot asBatch() {
        List<Tensor> values = this.values();
        List<FieldVector> vectors = new ArrayList<>();
        Schema schema = this.schema();
        for (int index = 0; index < this.columnCount; ++index) {
            val array = values.get(index).getArray();
            vectors.add(array);
        }
        return new VectorSchemaRoot(schema, vectors, rowCount);
    }
}

class DataFrameResolver extends ObjectFactory.Resolver {
    @SneakyThrows(VineyardException.class)
    @Override
    public Object resolve(final ObjectMeta meta) {
        val columns = ImmutableList.copyOf(meta.getArrayValue("columns_"));
        val columnCount = columns.size();
        val values =
                IntStream.range(0, columnCount)
                        .mapToObj(index -> (Tensor) meta.getMember("__values_-value-" + index))
                        .collect(Collectors.toList());
        val rowCount = values.get(0).getArray().getValueCount();
        return new DataFrame(meta, rowCount, columnCount, columns, values);
    }
}
