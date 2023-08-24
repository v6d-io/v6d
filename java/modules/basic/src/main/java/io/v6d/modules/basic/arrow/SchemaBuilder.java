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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.v6d.core.client.Client;
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

    private byte[] buffer;

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
        this.buffer = SchemaSerializer.serialize(new Schema(fields, customMetadata));
    }

    @SneakyThrows(JsonProcessingException.class)
    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::SchemaProxy");
        meta.setNBytes(buffer.length);
        val mapper = new ObjectMapper();
        val schema_binary = mapper.createObjectNode();
        val array = mapper.createArrayNode();
        for (val item : buffer) {
            array.add(Byte.toUnsignedInt(item));
        }
        schema_binary.put("bytes", array);
        meta.setValue("schema_binary_", mapper.writeValueAsString(schema_binary));
        return client.createMetaData(meta);
    }
}
