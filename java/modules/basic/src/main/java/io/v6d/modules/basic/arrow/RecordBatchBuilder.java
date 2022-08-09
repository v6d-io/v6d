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

import static java.util.Objects.requireNonNull;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.columnar.ColumnarDataBuilder;
import java.util.ArrayList;
import java.util.List;
import lombok.*;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RecordBatchBuilder implements ObjectBuilder {
    private Logger logger = LoggerFactory.getLogger(RecordBatchBuilder.class);

    private long rows;
    private final SchemaBuilder schemaBuilder;
    private List<ArrayBuilder> arrayBuilders;
    private List<ColumnarDataBuilder> columnBuilders;

    private boolean schemaMutable = true;

    public RecordBatchBuilder(final IPCClient client, int rows) {
        this.rows = rows;
        schemaBuilder = new SchemaBuilder();
    }

    public RecordBatchBuilder(final IPCClient client, final Schema schema, int rows)
            throws VineyardException {
        this.rows = rows;
        schemaBuilder = SchemaBuilder.fromSchema(schema);
        this.finishSchema(client);
    }

    public RecordBatchBuilder(
            final IPCClient client, final Schema schema, List<ColumnarDataBuilder> columnBuilders)
            throws VineyardException {
        this.schemaBuilder = SchemaBuilder.fromSchema(schema);
        this.schemaMutable = false;

        this.columnBuilders = requireNonNull(columnBuilders, "column builders is null");
    }

    public void addField(final Field field) throws VineyardException {
        VineyardException.asserts(schemaMutable, "cannot continue to add columns");
        this.schemaBuilder.addField(field);
    }

    public void addCustomMetadata(String key, String value) {
        this.schemaBuilder.addMetadata(key, value);
    }

    public void finishSchema(IPCClient client) throws VineyardException {
        this.schemaMutable = false;

        if (arrayBuilders != null && columnBuilders != null) {
            return;
        }

        // generate builders
        arrayBuilders = new ArrayList<>();
        columnBuilders = new ArrayList<>();
        for (val field : schemaBuilder.getFields()) {
            val builder = arrayBuilderFor(client, field);
            arrayBuilders.add(builder);
            columnBuilders.add(builder.columnar());
        }
    }

    public List<ColumnarDataBuilder> getColumnBuilders() throws VineyardException {
        VineyardException.asserts(!schemaMutable, "the schema builder is not finished yet");
        return columnBuilders;
    }

    public ColumnarDataBuilder getColumnBuilder(int index) throws VineyardException {
        VineyardException.asserts(!schemaMutable, "the schema builder is not finished yet");
        return columnBuilders.get(index);
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);

        val meta = ObjectMeta.empty();

        meta.setTypename("vineyard::RecordBatch");
        meta.setValue("column_num_", arrayBuilders.size());
        meta.setValue("row_num_", rows);
        meta.setValue("__columns_-size", arrayBuilders.size());
        meta.addMember("schema_", schemaBuilder.seal(client));

        for (int index = 0; index < arrayBuilders.size(); ++index) {
            meta.addMember("__columns_-" + index, arrayBuilders.get(index).seal(client));
        }
        meta.setNBytes(0); // FIXME

        return client.createMetaData(meta);
    }

    private ArrayBuilder arrayBuilderFor(IPCClient client, Field field) throws VineyardException {
        if (field.getType().equals(Arrow.Type.Boolean)) {
            return new BooleanArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Int)) {
            return new Int32ArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Int64)) {
            return new Int64ArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Float)) {
            return new FloatArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Double)) {
            return new DoubleArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.VarChar)) {
            return new StringArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Null)) {
            return new NullArrayBuilder(client, rows);
        } else {
            throw new VineyardException.NotImplemented(
                    "array builder for type " + field.getType() + " is not supported");
        }
    }
}
