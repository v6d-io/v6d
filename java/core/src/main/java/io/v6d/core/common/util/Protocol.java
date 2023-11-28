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
package io.v6d.core.common.util;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.v6d.core.common.memory.Payload;
import java.util.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.val;

public class Protocol {
    public abstract static class Request {
        protected static void check(JsonNode tree, String type) throws VineyardException {
            VineyardException.AssertionFailed.AssertEqual(type, tree.get("type").asText());
        }
    }

    public abstract static class Reply {
        protected static void check(JsonNode tree, String type) throws VineyardException {
            if (tree.has("code")) {
                VineyardException.check(JSON.getInt(tree, "code"), tree.get("message").asText(""));
            }
            VineyardException.AssertionFailed.AssertEqual(type, JSON.getText(tree, "type"));
        }

        public abstract void get(JsonNode root) throws VineyardException;
    }

    public static class RegisterRequest extends Request {
        public static void put(ObjectNode root) {
            root.put("type", "register_request");
            root.put("version", "0.0.0"); // FIXME
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class RegisterReply extends Reply {
        private String ipc_socket;
        private String rpc_endpoint;
        private InstanceID instance_id;
        private String version;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "register_reply");
            this.ipc_socket = JSON.getText(root, "ipc_socket");
            this.rpc_endpoint = JSON.getText(root, "rpc_endpoint");
            this.instance_id = new InstanceID(JSON.getLong(root, "instance_id"));
            this.version = root.get("version").asText("0.0.0");
        }
    }

    public static class CreateDataRequest extends Request {
        public static void put(ObjectNode root, ObjectNode content) {
            root.put("type", "create_data_request");
            root.set("content", content);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class CreateDataReply extends Reply {
        private ObjectID id;
        private Signature signature;
        private InstanceID instance_id;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "create_data_reply");
            this.id = new ObjectID(JSON.getLong(root, "id"));
            this.signature = new Signature(JSON.getLong(root, "signature"));
            this.instance_id = new InstanceID(JSON.getLong(root, "instance_id"));
        }
    }

    public static class GetDataRequest extends Request {
        public static void put(ObjectNode root, ObjectID id, boolean sync_remote, boolean wait) {
            root.put("type", "get_data_request");
            ObjectMapper mapper = new ObjectMapper();
            val ids = mapper.createArrayNode();
            ids.add(id.value());
            root.put("id", ids);
            root.put("sync_remote", sync_remote);
            root.put("wait", wait);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetDataReply extends Reply {
        private Map<ObjectID, ObjectNode> contents;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "get_data_reply");
            this.contents = new HashMap<>();
            val fields = root.get("content").fields();
            while (fields.hasNext()) {
                val field = fields.next();
                this.contents.put(
                        ObjectID.fromString(field.getKey()), (ObjectNode) field.getValue());
            }
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class ListNameReply extends Reply {
        private Map<String, ObjectID> contents;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "list_name_reply");
            this.contents = new HashMap<>();
            val fields = root.get("names").fields();
            while (fields.hasNext()) {
                val field = fields.next();
                this.contents.put(field.getKey(), new ObjectID(field.getValue().asLong()));
            }
        }
    }

    public static class CreateBufferRequest extends Request {
        public static void put(ObjectNode root, long size) {
            root.put("type", "create_buffer_request");
            root.put("size", size);
        }
    }

    public static class SealRequest extends Request {
        public static void put(ObjectNode root, ObjectID id) {
            root.put("type", "seal_request");
            root.put("object_id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class CreateBufferReply extends Reply {
        private ObjectID id;
        private Payload payload;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "create_buffer_reply");
            this.id = new ObjectID(JSON.getLong(root, "id"));
            this.payload = Payload.fromJson(root.get("created"));
        }
    }

    public static class SealReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "seal_reply");
        }
    }

    public static class GetBuffersRequest extends Request {
        public static void put(ObjectNode root, List<ObjectID> ids) {
            root.put("type", "get_buffers_request");
            int index = 0;
            for (val id : ids) {
                root.put(String.valueOf(index++), id.value());
            }
            root.put("num", ids.size());
        }

        public static void put(ObjectNode root, Set<ObjectID> ids) {
            root.put("type", "get_buffers_request");
            int index = 0;
            for (val id : ids) {
                root.put(String.valueOf(index++), id.value());
            }
            root.put("num", ids.size());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetBuffersReply extends Reply {
        private List<Payload> payloads;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "get_buffers_reply");
            this.payloads = new ArrayList<>();
            for (int index = 0; index < JSON.getInt(root, "num"); ++index) {
                val payload = Payload.fromJson(root.get(String.valueOf(index)));
                this.payloads.add(payload);
            }
        }
    }

    public static class SealBufferRequest extends Request {
        public static void put(ObjectNode root, ObjectID id) {
            root.put("type", "seal_request");
            root.put("object_id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class SealBufferReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "seal_reply");
        }
    }

    public static class ShrinkBufferRequest extends Request {
        public static void put(ObjectNode root, ObjectID id, long size) {
            root.put("type", "shrink_buffer_request");
            root.put("id", id.value());
            root.put("size", size);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class ShrinkBufferReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "shrink_buffer_reply");
        }
    }

    public static class ListDataRequest extends Request {
        public static void put(ObjectNode root, String pattern, boolean regex, int limit) {
            root.put("type", "list_data_request");
            root.put("pattern", pattern);
            root.put("regex", regex);
            root.put("limit", limit);
        }
    }

    public static class ListNameRequest extends Request {
        public static void put(ObjectNode root, String pattern, boolean regex, int limit) {
            root.put("type", "list_name_request");
            root.put("pattern", pattern);
            root.put("regex", regex);
            root.put("limit", limit);
        }
    }

    public static class PersistRequest extends Request {
        public static void put(ObjectNode root, ObjectID id) {
            root.put("type", "persist_request");
            root.put("id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class PersistReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "persist_reply");
        }
    }

    public static class DeleteDataRequest extends Request {
        public static void put(
                ObjectNode root,
                Collection<ObjectID> ids,
                boolean force,
                boolean deep,
                boolean fastpath) {
            root.put("type", "del_data_request");
            val array = root.putArray("id");
            ids.forEach(id -> array.add(id.value()));
            root.put("force", force);
            root.put("deep", deep);
            root.put("fastpath", fastpath);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class DeleteDataReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "del_data_reply");
        }
    }

    public static class MigrateObjectRequest extends Request {
        public static void put(
                ObjectNode root,
                ObjectID id,
                boolean local,
                boolean isStream,
                String peer,
                String peerRpcEndpoint) {
            root.put("type", "migrate_object_request");
            root.put("object_id", id.value());
            root.put("local", local);
            root.put("is_stream", isStream);
            root.put("peer", peer);
            root.put("peer_rpc_endpoint", peerRpcEndpoint);
        }

        public static void put(ObjectNode root, ObjectID id) {
            root.put("type", "migrate_object_request");
            root.put("object_id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class MigrateObjectReply extends Reply {
        private ObjectID objectID;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "migrate_object_reply");
            objectID = new ObjectID(JSON.getLong(root, "object_id"));
        }
    }

    public static class PutNameRequest extends Request {
        public static void put(ObjectNode root, ObjectID id, String name) {
            root.put("type", "put_name_request");
            root.put("object_id", id.value());
            root.put("name", name);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class PutNameReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "put_name_reply");
        }
    }

    public static class GetNameRequest extends Request {
        public static void put(ObjectNode root, String name, boolean wait) {
            root.put("type", "get_name_request");
            root.put("name", name);
            root.put("wait", wait);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetNameReply extends Reply {
        private ObjectID id;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "get_name_reply");
            this.id = new ObjectID(JSON.getLong(root, "object_id"));
        }
    }

    public static class DropNameRequest extends Request {
        public static void put(ObjectNode root, String name) {
            root.put("type", "drop_name_request");
            root.put("name", name);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class DropNameReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "drop_name_reply");
        }
    }

    public static class CreateStreamRequest extends Request {
        public static void put(ObjectNode root, final ObjectID id) {
            root.put("type", "create_stream_request");
            root.put("object_id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class CreateStreamReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "create_stream_reply");
        }
    }

    public static class OpenStreamRequest extends Request {
        public static void put(ObjectNode root, final ObjectID id, final int mode) {
            root.put("type", "open_stream_request");
            root.put("object_id", id.value());
            root.put("mode", mode);
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class OpenStreamReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "open_stream_reply");
        }
    }

    public static class PushNextStreamChunkRequest extends Request {
        public static void put(ObjectNode root, final ObjectID id, final ObjectID chunk) {
            root.put("type", "push_next_stream_chunk_request");
            root.put("id", id.value());
            root.put("chunk", chunk.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class PushNextStreamChunkReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "push_next_stream_chunk_reply");
        }
    }

    public static class PullNextStreamChunkRequest extends Request {
        public static void put(ObjectNode root, final ObjectID id) {
            root.put("type", "pull_next_stream_chunk_request");
            root.put("id", id.value());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class PullNextStreamChunkReply extends Reply {
        private ObjectID chunk;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "pull_next_stream_chunk_reply");
            this.chunk = new ObjectID(JSON.getLong(root, "chunk"));
        }
    }

    public static class StopStreamRequest extends Request {
        public static void put(ObjectNode root, final ObjectID id, final boolean failed) {
            root.put("type", "stop_stream_request");
            root.put("id", id.value());
            root.put("failed", failed);
        }
    }

    public static class InstanceStatusRequest extends Request {
        public static void put(ObjectNode root) {
            root.put("type", "instance_status_request");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class InstanceStatusReply extends Reply {
        private ObjectNode status;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "instance_status_reply");
            this.status = (ObjectNode) root.get("meta");
        }
    }

    public static class ClusterStatusRequest extends Request {
        public static void put(ObjectNode root) {
            root.put("type", "cluster_meta");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class ClusterStatusReply extends Reply {
        private ObjectNode status;

        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "cluster_meta");
            this.status = (ObjectNode) root.get("meta");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class StopStreamReply extends Reply {
        @Override
        public void get(JsonNode root) throws VineyardException {
            check(root, "stop_stream_reply");
        }
    }
}
