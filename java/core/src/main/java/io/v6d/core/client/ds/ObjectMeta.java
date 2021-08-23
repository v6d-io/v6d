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
package io.v6d.core.client.ds;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.v6d.core.common.util.InstanceID;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.Signature;
import io.v6d.core.common.util.VineyardException;
import lombok.SneakyThrows;
import lombok.val;

public class ObjectMeta {
    private ObjectMapper mapper;
    private ObjectNode meta;
    private InstanceID instanceID;
    private BufferSet buffers;

    private ObjectMeta() {
        mapper = new ObjectMapper();
        meta = mapper.createObjectNode();
        buffers = new BufferSet();
    }

    public static ObjectMeta fromMeta(ObjectNode meta, InstanceID instanceID)
            throws VineyardException {
        val metadata = new ObjectMeta();
        metadata.meta = meta;
        metadata.instanceID = instanceID;
        metadata.findAllBuffers(meta);
        return metadata;
    }

    public BufferSet getBuffers() {
        return this.buffers;
    }

    public void reset() {
        this.meta = mapper.createObjectNode();
    }

    public void setBuffer(ObjectID id, Buffer buffer) throws VineyardException {
        if (!buffers.contains(id)) {
            throw new VineyardException.AssertionFailed("Buffer not found for " + id);
        }
        buffers.emplace(id, buffer);
    }

    public Buffer getBuffer(ObjectID id) {
        return this.buffers.get(id);
    }

    public ObjectNode metadata() {
        return this.meta;
    }

    @SneakyThrows(JsonProcessingException.class)
    public String metadataString() {
        return mapper.writeValueAsString(this.meta);
    }

    public ObjectID id() {
        return ObjectID.fromString(this.getStringValue("id"));
    }

    public Signature signature() {
        return new Signature(this.getLongValue("signature"));
    }

    public InstanceID instance_id() {
        return new InstanceID(this.getLongValue("instance_id"));
    }

    public boolean isGlobal() {
        return this.getBooleanValue("global");
    }

    public String typename() {
        return this.getStringValue("typename");
    }

    public long nbytes() {
        return this.getLongValue("nbytes");
    }

    public String getStringValue(String key) {
        return this.meta.get(key).asText();
    }

    public int getIntValue(String key) {
        return this.meta.get(key).asInt();
    }

    public long getLongValue(String key) {
        return this.meta.get(key).asLong();
    }

    public float getFloatValue(String key) {
        return (float) this.meta.get(key).asDouble();
    }

    public double getDoubleValue(String key) {
        return this.meta.get(key).asDouble();
    }

    public boolean getBooleanValue(String key) {
        return this.meta.get(key).asBoolean();
    }

    public JsonNode getValue(String key) {
        return this.meta.get(key);
    }

    @Override
    public String toString() {
        return "ObjectMeta{"
                + "meta="
                + meta
                + ", instanceID="
                + instanceID
                + ", buffers="
                + buffers
                + '}';
    }

    private void findAllBuffers(ObjectNode meta) throws VineyardException {
        if (meta == null || meta.isEmpty()) {
            return;
        }
        ObjectID member = ObjectID.fromString(meta.get("id").asText());
        if (member.isBlob()) {
            if (meta.get("instance_id").asLong() == this.instanceID.Value()) {
                this.buffers.emplace(member);
            }
        } else {
            val fields = meta.fields();
            while (fields.hasNext()) {
                val item = fields.next().getValue();
                if (item.isObject()) {
                    this.findAllBuffers((ObjectNode) item);
                }
            }
        }
    }
}
