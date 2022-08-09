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
package io.v6d.core.client.ds;

import static com.google.common.base.MoreObjects.toStringHelper;
import static java.util.Objects.requireNonNull;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.v6d.core.common.util.InstanceID;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.Signature;
import io.v6d.core.common.util.VineyardException;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import lombok.SneakyThrows;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ObjectMeta {
    private static final Logger logger = LoggerFactory.getLogger(ObjectMeta.class);

    private ObjectMapper mapper;
    private ObjectNode meta;
    private InstanceID instanceId;
    private BufferSet buffers;
    private boolean incomplete;

    private ObjectMeta() {
        mapper = new ObjectMapper();
        meta = mapper.createObjectNode();
        instanceId = InstanceID.UnspecifiedInstanceID;
        buffers = new BufferSet();
        incomplete = false;

        this.setLocal();
        this.setTransient();
    }

    public static ObjectMeta fromMeta(ObjectNode meta, InstanceID instanceId)
            throws VineyardException {
        val metadata = new ObjectMeta();
        metadata.meta = requireNonNull(meta, "meta is null");
        metadata.instanceId = instanceId;
        metadata.findAllBuffers(meta);
        return metadata;
    }

    public static ObjectMeta empty() {
        return new ObjectMeta();
    }

    public BufferSet getBuffers() {
        return this.buffers;
    }

    public void reset() {
        this.meta = mapper.createObjectNode();
    }

    public void replace(final ObjectMeta meta) {
        this.meta = meta.meta;
        this.instanceId = meta.getInstanceId();
        this.buffers = meta.getBuffers();
        this.incomplete = false;
    }

    public void setBuffer(ObjectID id, Buffer buffer) throws VineyardException {
        if (!buffers.contains(id)) {
            throw new VineyardException.AssertionFailed("Buffer not found for " + id);
        }
        buffers.emplace(id, buffer);
    }

    public void setBufferUnchecked(ObjectID id, Buffer buffer) throws VineyardException {
        buffers.emplaceUnchecked(id, buffer);
    }

    public Buffer getBuffer(ObjectID id) {
        return this.buffers.get(id);
    }

    public Buffer getBufferChecked(ObjectID id) {
        val buffer = this.buffers.get(id);
        if (buffer == null) {
            throw new NullPointerException("buffer not found for " + id.toString());
        }
        return buffer;
    }

    @SneakyThrows(VineyardException.class)
    public ObjectMeta getMemberMeta(String name) {
        if (!this.meta.has(name)) {
            throw new VineyardException.AssertionFailed(
                    "Failed to get member: " + name + ", no such member");
        }
        if (!this.meta.get(name).isObject()) {
            throw new VineyardException.AssertionFailed(
                    "Failed to get member: " + name + ", member field is not object");
        }
        val ret = ObjectMeta.fromMeta((ObjectNode) this.meta.get(name), this.instanceId);
        val all_blobs = buffers.allBuffers();
        val blobs_to_fill = new HashMap<ObjectID, Buffer>();
        for (val blob : ret.buffers.allBuffers().entrySet()) {
            if (all_blobs.containsKey(blob.getKey())) {
                blobs_to_fill.put(blob.getKey(), all_blobs.get(blob.getKey()));
            }
        }
        for (val blob : blobs_to_fill.entrySet()) {
            ret.setBuffer(blob.getKey(), blob.getValue());
        }
        return ret;
    }

    public void addMemberMeta(String name, ObjectMeta meta) {
        this.meta.set(name, meta.metadata());
        this.buffers.extend(meta.buffers);
    }

    public Object getMember(String name) {
        return ObjectFactory.getFactory().resolve(getMemberMeta(name));
    }

    public void addMember(String name, Object value) {
        this.addMemberMeta(name, value.getMeta());
    }

    public void addMember(String name, ObjectMeta value) {
        this.addMemberMeta(name, value);
    }

    public void addMember(String name, ObjectID objectID) {
        val placeholder = mapper.createObjectNode();
        placeholder.put("id", objectID.toString());
        this.meta.set(name, placeholder);
        incomplete = true;
    }

    public ObjectNode metadata() {
        return this.meta;
    }

    @SneakyThrows(JsonProcessingException.class)
    public String metadataString() {
        return mapper.writeValueAsString(this.meta);
    }

    public boolean isIncomplete() {
        return this.incomplete;
    }

    public ObjectID getId() {
        return ObjectID.fromString(this.getStringValue("id"));
    }

    public void setId(ObjectID id) {
        this.meta.put("id", id.toString());
    }

    public boolean hasName() {
        return this.meta.has("__name");
    }

    public String getName() {
        return this.meta.get("__name").textValue();
    }

    public Signature getSignature() {
        return new Signature(this.getLongValue("signature"));
    }

    public void setSignature(Signature signature) {
        this.meta.put("signature", signature.Value());
    }

    public InstanceID getInstanceId() {
        return new InstanceID(this.getLongValue("instance_id"));
    }

    public void setInstanceId(final InstanceID instanceId) {
        this.instanceId = instanceId;
        this.meta.put("instance_id", instanceId.value());
    }

    public boolean isGlobal() {
        return this.getBooleanValue("global");
    }

    public void setGlobal() {
        this.setGlobal(true);
    }

    public void setLocal() {
        this.setGlobal(false);
    }

    public void setGlobal(boolean global) {
        this.meta.put("global", global);
    }

    public boolean isPersist() {
        return !this.getBooleanValue("transient");
    }

    public void setTransient() {
        this.setTransient(true);
    }

    public void setPersist() {
        this.setTransient(false);
    }

    public void setTransient(boolean transient_) {
        this.meta.put("transient", transient_);
    }

    public String getTypename() {
        return this.getStringValue("typename");
    }

    public void setTypename(final String typename) {
        this.meta.put("typename", typename);
    }

    public long getNBytes() {
        return this.getLongValue("nbytes");
    }

    public void setNBytes(long nbytes) {
        this.meta.put("nbytes", nbytes);
    }

    public String getStringValue(String key) {
        return this.meta.get(key).textValue();
    }

    public void setValue(String key, String value) {
        this.meta.put(key, value);
    }

    public int getIntValue(String key) {
        return this.meta.get(key).intValue();
    }

    public void setValue(String key, int value) {
        this.meta.put(key, value);
    }

    public long getLongValue(String key) {
        return this.meta.get(key).longValue();
    }

    public void setValue(String key, long value) {
        this.meta.put(key, value);
    }

    public float getFloatValue(String key) {
        return (float) this.meta.get(key).floatValue();
    }

    public void setValue(String key, float value) {
        this.meta.put(key, value);
    }

    public double getDoubleValue(String key) {
        return this.meta.get(key).doubleValue();
    }

    public void setValue(String key, double value) {
        this.meta.put(key, value);
    }

    public boolean getBooleanValue(String key) {
        return this.meta.get(key).booleanValue();
    }

    public void setValue(String key, boolean value) {
        this.meta.put(key, value);
    }

    @SneakyThrows(IOException.class)
    public <T> Collection<T> getListValue(String key) {
        val value = getValue(key);
        assert value.isArray();
        val reader = mapper.readerFor(new TypeReference<List<T>>() {});
        return reader.readValue(value);
    }

    @SneakyThrows
    public <T> void setListValue(String key, Collection<T> values) {
        val array = this.meta.putArray(key);
        for (val value : values) {
            array.add(mapper.valueToTree(value));
        }
    }

    public JsonNode getValue(String key) {
        return this.meta.get(key);
    }

    public void setValue(String key, JsonNode value) {
        this.meta.set(key, value);
    }

    public ObjectNode getDictValue(String key) {
        // FIXME
        val value = getValue(key);
        assert value.isObject();
        return (ObjectNode) value;
    }

    public void setDictNode(String key, ObjectNode values) {
        // FIXME
        this.meta.set(key, values);
    }

    public ArrayNode getArrayValue(String key) {
        val value = getValue(key);
        assert value.isObject();
        return (ArrayNode) value;
    }

    public void setArrayNode(String key, ArrayNode values) {
        this.meta.set(key, values);
    }

    public boolean has(String key) {
        return this.meta.has(key);
    }

    public boolean hasMeta(String key) {
        return this.meta.has(key) && !this.meta.get(key).isObject();
    }

    public boolean hasMember(String key) {
        return this.meta.has(key) && this.meta.get(key).isObject();
    }

    @Override
    public String toString() {
        return toStringHelper(this)
                .add("instance_id", instanceId)
                .add("meta", meta)
                .add("buffers", buffers)
                .toString();
    }

    @SneakyThrows(JsonProcessingException.class)
    public String toPrettyString() {
        return mapper.writerWithDefaultPrettyPrinter().writeValueAsString(meta);
    }

    private void findAllBuffers(ObjectNode meta) throws VineyardException {
        if (meta == null || meta.isEmpty()) {
            return;
        }
        ObjectID member = ObjectID.fromString(meta.get("id").textValue());
        if (member.isBlob()) {
            if (meta.get("instance_id").asLong() == this.instanceId.value()) {
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
