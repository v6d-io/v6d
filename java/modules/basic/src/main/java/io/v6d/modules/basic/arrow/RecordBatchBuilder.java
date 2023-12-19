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

import static java.util.Objects.requireNonNull;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.util.ObjectTransformer;
import io.v6d.modules.basic.columnar.ColumnarDataBuilder;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;
import lombok.*;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
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
    private ObjectTransformer transformers;

    private boolean schemaMutable = true;

    public RecordBatchBuilder(final IPCClient client, int rows) {
        this.rows = rows;
        schemaBuilder = new SchemaBuilder();
    }

    public RecordBatchBuilder(final IPCClient client, final Schema schema, int rows)
            throws VineyardException {
        this(client, schema, rows, new ObjectTransformer());
    }

    public RecordBatchBuilder(
            final IPCClient client, final Schema schema, int rows, ObjectTransformer transformers)
            throws VineyardException {
        this.rows = rows;
        schemaBuilder = SchemaBuilder.fromSchema(schema);
        this.transformers = transformers;
        this.finishSchema(client);
    }

    public RecordBatchBuilder(
            final IPCClient client,
            final Schema schema,
            List<ArrayBuilder> arrayBuilders,
            List<ColumnarDataBuilder> columnBuilders,
            int rows)
            throws VineyardException {
        this.schemaBuilder = SchemaBuilder.fromSchema(schema);
        this.schemaMutable = false;
        this.rows = rows;

        // Fill array builder in the future.
        this.arrayBuilders = requireNonNull(arrayBuilders, "array builders is null");
        this.columnBuilders = requireNonNull(columnBuilders, "column builders is null");
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
            ColumnarDataBuilder columnarDataBuilder = builder.columnar(transformers);
            columnBuilders.add(columnarDataBuilder);
        }
    }

    public void addField(final Field field) throws VineyardException {
        VineyardException.asserts(schemaMutable, "cannot continue to add columns");
        this.schemaBuilder.addField(field);
    }

    public void addCustomMetadata(String key, String value) {
        this.schemaBuilder.addMetadata(key, value);
    }

    public List<ColumnarDataBuilder> getColumnBuilders() throws VineyardException {
        VineyardException.asserts(!schemaMutable, "the schema builder is not finished yet");
        return columnBuilders;
    }

    public ColumnarDataBuilder getColumnBuilder(int column_index) throws VineyardException {
        VineyardException.asserts(!schemaMutable, "the schema builder is not finished yet");
        return columnBuilders.get(column_index);
    }

    public long getNumRows() {
        return this.rows;
    }

    public long getNumColumns() {
        return this.columnBuilders.size();
    }

    public void shrink(Client client, long size) throws VineyardException {
        for (val builder : arrayBuilders) {
            builder.shrink(client, size);
        }
        this.rows = size;
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

        // Currently each vineyardd and each namenode manager are on the same node.
        // FIXME: This property does not seem to work in yarn.
        InetAddress localhost;
        String hostname;
        try {
            localhost = InetAddress.getLocalHost();
            hostname = localhost.getHostName();
        } catch (Exception e) {
            hostname = "unknown";
        }
        meta.setValue("host_", hostname);

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
        } else if (field.getType().equals(Arrow.Type.LargeVarChar)) {
            return new LargeStringArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.VarBinary)) {
            return new VarBinaryArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.LargeVarBinary)) {
            return new LargeStringArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Null)) {
            return new NullArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.TinyInt)) {
            return new Int8ArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.SmallInt)) {
            return new Int16ArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.Date)) {
            return new DateArrayBuilder(client, rows);
        } else if (field.getType().equals(Arrow.Type.TimeStampMicro)) {
            return new TimestampArrayBuilder(client, rows, TimeUnit.MICROSECOND);
        } else if (field.getType().equals(Arrow.Type.TimeStampMilli)) {
            return new TimestampArrayBuilder(client, rows, TimeUnit.MILLISECOND);
        } else if (field.getType().equals(Arrow.Type.TimeStampNano)) {
            return new TimestampArrayBuilder(client, rows, TimeUnit.NANOSECOND);
        } else if (field.getType().equals(Arrow.Type.TimeStampSec)) {
            return new TimestampArrayBuilder(client, rows, TimeUnit.SECOND);
        } else if (field.getType() instanceof ArrowType.Decimal) {
            ArrowType.Decimal decimal = (ArrowType.Decimal) field.getType();
            return new DecimalArrayBuilder(
                    client,
                    rows,
                    decimal.getPrecision(),
                    decimal.getScale(),
                    decimal.getBitWidth());
        } else if (field.getType().equals(Arrow.Type.List)) {
            return new ListArrayBuilder(client, field);
        } else if (field.getType().equals(Arrow.Type.Struct)) {
            return new StructArrayBuilder(client, field);
        } else if (field.getType().equals(Arrow.Type.Map)) {
            return new MapArrayBuilder(client, field);
        } else {
            throw new VineyardException.NotImplemented(
                    "array builder for type " + field.getType() + " is not supported");
        }
    }
}
