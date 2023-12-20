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
import io.v6d.core.client.ds.ObjectBase;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import lombok.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TableBuilder implements ObjectBuilder {
    private Logger logger = LoggerFactory.getLogger(TableBuilder.class);

    private final SchemaBuilder schemaBuilder;
    private final List<ObjectBase> batches;

    public TableBuilder(
            final Client client,
            final SchemaBuilder schemaBuilder,
            final List<RecordBatchBuilder> batchBuilders) {
        this.schemaBuilder = requireNonNull(schemaBuilder, "schema is null");
        this.batches = new ArrayList<>(requireNonNull(batchBuilders, "batches are null"));
    }

    public TableBuilder(final Client client, final SchemaBuilder schemaBuilder) {
        this.schemaBuilder = requireNonNull(schemaBuilder, "schema is null");
        this.batches = new LinkedList<>();
    }

    public void addBatch(RecordBatch batch) {
        this.batches.add(batch);
    }

    public void addBatch(RecordBatchBuilder builder) {
        this.batches.add(builder);
    }

    public List<ObjectBase> getBatches() {
        return this.batches;
    }

    public int getBatchSize() {
        return this.batches.size();
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);

        val meta = ObjectMeta.empty();

        meta.setTypename("vineyard::Table");
        meta.setValue("batch_num_", batches.size());
        meta.setValue("num_rows_", -1);
        meta.setValue("num_columns_", schemaBuilder.getFields().size());
        meta.addMember("schema_", schemaBuilder.seal(client));
        meta.setGlobal(true);

        meta.setValue("partitions_-size", batches.size());
        for (int index = 0; index < batches.size(); ++index) {
            meta.addMember("partitions_-" + index, batches.get(index).seal(client));
        }
        meta.setNBytes(0); // FIXME

        return client.createMetaData(meta);
    }

    public static ObjectMeta fromRecordBatchMeta(
            Client client, ObjectMeta schemaMeta, List<ObjectMeta> recordBatchMetas, int columns)
            throws VineyardException {
        val meta = ObjectMeta.empty();

        meta.setTypename("vineyard::Table");
        meta.setValue("batch_num_", recordBatchMetas.size());
        meta.setValue("num_rows_", -1);
        meta.setValue("num_columns_", columns);
        meta.addMember("schema_", schemaMeta);

        meta.setValue("partitions_-size", recordBatchMetas.size());
        for (int index = 0; index < recordBatchMetas.size(); ++index) {
            meta.addMember("partitions_-" + index, recordBatchMetas.get(index));
        }
        meta.setGlobal(true);
        meta.setNBytes(0);
        return client.createMetaData(meta);
    }

    public static ObjectMeta mergeTables(Client client, ObjectMeta[] tableMetas)
            throws VineyardException {
        if (tableMetas.length == 0) {
            return null;
        }

        List<ObjectMeta> recordBatchMetas = new ArrayList<>();
        ObjectMeta schemaMeta = tableMetas[0].getMemberMeta("schema_");
        int columns = 0;

        for (int i = 0; i < tableMetas.length; i++) {
            columns += tableMetas[i].getIntValue("num_columns_");
            int batchNum = tableMetas[i].getIntValue("batch_num_");
            for (int j = 0; j < batchNum; j++) {
                recordBatchMetas.add(tableMetas[i].getMemberMeta("partitions_-" + j));
            }
        }
        return fromRecordBatchMeta(client, schemaMeta, recordBatchMetas, columns);
    }
}
