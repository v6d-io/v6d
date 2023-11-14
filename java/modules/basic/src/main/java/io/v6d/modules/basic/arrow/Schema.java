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

import static com.google.common.base.MoreObjects.toStringHelper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.google.common.base.Objects;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.util.SchemaSerializer;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import lombok.SneakyThrows;
import lombok.val;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Collections2;
import org.apache.arrow.vector.types.pojo.Field;

public class Schema extends Object implements Serializable {
    private org.apache.arrow.vector.types.pojo.Schema schema;

    public static void instantiate() {
        Buffer.instantiate();
        ObjectFactory.getFactory().register("vineyard::SchemaProxy", new SchemaResolver());
    }

    public Schema(final ObjectMeta meta, org.apache.arrow.vector.types.pojo.Schema schema) {
        super(meta);
        this.schema = schema;
    }

    public Schema(final ObjectMeta meta, List<Field> fields) {
        super(meta);
        this.schema =
                new org.apache.arrow.vector.types.pojo.Schema(
                        Collections2.immutableListCopy(fields));
    }

    public Schema(final ObjectMeta meta, List<Field> fields, Map<String, String> metadata) {
        super(meta);
        this.schema =
                new org.apache.arrow.vector.types.pojo.Schema(
                        Collections2.immutableListCopy(fields),
                        Collections2.immutableMapCopy(metadata));
    }

    public org.apache.arrow.vector.types.pojo.Schema getSchema() {
        return schema;
    }

    @Override
    public boolean equals(java.lang.Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Schema schema1 = (Schema) o;
        return Objects.equal(schema, schema1.schema);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(schema);
    }

    @Override
    public String toString() {
        return toStringHelper(this)
                .add("object", super.toString())
                .add("schema", schema)
                .toString();
    }

    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.writeObject(SchemaSerializer.serialize(schema));
    }

    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException {
        byte[] bytes = (byte[]) in.readObject();
        val allocator = new RootAllocator(Long.MAX_VALUE);
        val buffer = allocator.buffer(bytes.length);
        buffer.writeBytes(bytes);
        this.schema = SchemaSerializer.deserialize(buffer, allocator);
    }
}

class SchemaResolver extends ObjectFactory.Resolver {
    @Override
    @SneakyThrows(IOException.class)
    public Object resolve(final ObjectMeta meta) {
        val mapper = new ObjectMapper();
        val node = mapper.readTree(meta.getStringValue("schema_binary_"));
        val buffer = (ArrayNode) node.get("bytes");
        val bytes = new byte[buffer.size()];
        for (int i = 0; i < buffer.size(); ++i) {
            bytes[i] = (byte) buffer.get(i).asInt();
        }
        return new Schema(meta, SchemaSerializer.deserialize(bytes, Arrow.default_allocator));
    }
}
