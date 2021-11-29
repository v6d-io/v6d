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

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.util.SchemaSerializer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import lombok.*;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

public class SchemaBuilder implements ObjectBuilder {
    private List<Field> fields;
    private Map<String, String> customMetadata;

    private BufferBuilder buffer;

    public SchemaBuilder() {
        fields = new ArrayList<>();
        customMetadata = new TreeMap<>();
    }

    public static SchemaBuilder fromSchema(Schema schema) {
        val builder = new SchemaBuilder();
        builder.fields = schema.getFields();
        builder.customMetadata = schema.getCustomMetadata();
        return builder;
    }

    public static SchemaBuilder fromSchema(io.v6d.modules.basic.arrow.Schema schema) {
        // FIXME: should be able to reusing
        return fromSchema(schema.getSchema());
    }

    public void addField(final Field field) {
        fields.add(field);
    }

    public void addMetadata(final String key, final String value) {
        customMetadata.put(key, value);
    }

    public List<Field> getFields() {
        return fields;
    }

    public Map<String, String> getCustomMetadata() {
        return customMetadata;
    }

    @Override
    @SneakyThrows(IOException.class)
    public void build(Client client) throws VineyardException {
        val bytes = SchemaSerializer.serialize(new Schema(fields, customMetadata));
        this.buffer = BufferBuilder.fromByteArray((IPCClient) client, bytes);
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::SchemaProxy");
        meta.setNBytes(buffer.length());
        meta.addMember("buffer_", buffer.seal(client));
        return client.createMetaData(meta);
    }
}
