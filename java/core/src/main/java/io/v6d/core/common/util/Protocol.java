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
                VineyardException.check(
                        tree.get("code").intValue(), tree.get("message").asText(""));
            }
            VineyardException.AssertionFailed.AssertEqual(type, tree.get("type").textValue());
        }

        public abstract void get(JsonNode root) throws VineyardException;
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
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
            this.ipc_socket = root.get("ipc_socket").textValue();
            this.rpc_endpoint = root.get("rpc_endpoint").textValue();
            this.instance_id = new InstanceID(root.get("instance_id").longValue());
            this.version = root.get("version").asText("0.0.0");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
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
            this.id = new ObjectID(root.get("id").longValue());
            this.signature = new Signature(root.get("signature").longValue());
            this.instance_id = new InstanceID(root.get("instance_id").longValue());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
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

    @EqualsAndHashCode(callSuper = false)
    public static class CreateBufferRequest extends Request {
        public static void put(ObjectNode root, long size) {
            root.put("type", "create_buffer_request");
            root.put("size", size);
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
            this.id = new ObjectID(root.get("id").longValue());
            this.payload = Payload.fromJson(root.get("created"));
        }
    }

    @EqualsAndHashCode(callSuper = false)
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
            for (int index = 0; index < root.get("num").intValue(); ++index) {
                val payload = Payload.fromJson(root.get(String.valueOf(index)));
                this.payloads.add(payload);
            }
        }
    }

    @EqualsAndHashCode(callSuper = false)
    public static class ListDataRequest extends Request {
        public static void put(ObjectNode root, String pattern, boolean regex, int limit) {
            root.put("type", "list_data_request");
            root.put("pattern", pattern);
            root.put("regex", regex);
            root.put("limit", limit);
        }
    }

    @Data
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

    @Data
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

    @Data
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
            this.id = new ObjectID(root.get("object_id").longValue());
        }
    }

    @Data
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
}
