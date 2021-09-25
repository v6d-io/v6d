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

        public abstract void Get(JsonNode root) throws VineyardException;
    }

    public abstract static class Reply {
        protected static void check(JsonNode tree, String type) throws VineyardException {
            if (tree.has("code")) {
                VineyardException.check(tree.get("code").asInt(), tree.get("message").asText(""));
            }
            VineyardException.AssertionFailed.AssertEqual(type, tree.get("type").asText());
        }

        public abstract void Get(JsonNode root) throws VineyardException;
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class RegisterRequest extends Request {
        private String version;

        public void Put(ObjectNode root) {
            root.put("type", "register_request");
            root.put("version", "0.0.0"); // FIXME
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "register_request");
            this.version = root.get("version").asText();
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class RegisterReply extends Reply {
        private String ipc_socket;
        private String rpc_endpoint;
        private InstanceID instance_id;
        private String version;

        public void Put(
                ObjectNode root, String ipc_socket, String rpc_endpoint, InstanceID instance_id) {
            root.put("type", "register_reply");
            root.put("ipc_socket", ipc_socket);
            root.put("rpc_endpoint", rpc_endpoint);
            root.put("instance_id", instance_id.Value());
            root.put("version", "0.0.0"); // FIXME
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "register_reply");
            this.ipc_socket = root.get("ipc_socket").asText();
            this.rpc_endpoint = root.get("rpc_endpoint").asText();
            this.instance_id = new InstanceID(root.get("instance_id").asLong());
            this.version = root.get("version").asText("0.0.0");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class CreateDataRequest extends Request {
        private JsonNode content;

        public void Put(ObjectNode root, ObjectNode content) {
            root.put("type", "create_data_request");
            root.put("content", content);
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "create_data_request");
            content = root.get("content");
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class CreateDataReply extends Reply {
        private ObjectID id;
        private Signature signature;
        private InstanceID instance_id;

        public void Put(ObjectNode root, ObjectID id, Signature signature, InstanceID instance_id) {
            root.put("type", "create_data_reply");
            root.put("id", id.Value());
            root.put("signature", signature.Value());
            root.put("instance_id", instance_id.Value());
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "create_data_reply");
            this.id = new ObjectID(root.get("id").asLong());
            this.signature = new Signature(root.get("signature").asLong());
            this.instance_id = new InstanceID(root.get("instance_id").asLong());
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetDataRequest extends Request {
        private ObjectID id;
        private boolean sync_remote;
        private boolean wait;

        public void Put(ObjectNode root, ObjectID id, boolean sync_remote, boolean wait) {
            root.put("type", "get_data_request");
            ObjectMapper mapper = new ObjectMapper();
            val ids = mapper.createArrayNode();
            ids.add(id.Value());
            root.put("id", ids);
            root.put("sync_remote", sync_remote);
            root.put("wait", wait);
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "get_data_request");
            this.id = new ObjectID(root.get("id").asLong());
            this.sync_remote = root.get("sync_remote").asBoolean();
            this.wait = root.get("wait").asBoolean();
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetDataReply extends Reply {
        private Map<ObjectID, ObjectNode> contents;

        public void Put(ObjectNode root, JsonNode content) {
            root.put("type", "get_data_reply");
            root.put("content", content);
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
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
    public static class GetBuffersRequest extends Request {
        private List<ObjectID> ids;

        public void Put(ObjectNode root, List<ObjectID> ids) {
            root.put("type", "get_buffers_request");
            int index = 0;
            for (val id : ids) {
                root.put(String.valueOf(index++), id.Value());
            }
            root.put("num", ids.size());
        }

        public void Put(ObjectNode root, Set<ObjectID> ids) {
            root.put("type", "get_buffers_request");
            int index = 0;
            for (val id : ids) {
                root.put(String.valueOf(index++), id.Value());
            }
            root.put("num", ids.size());
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "get_data_request");
            this.ids = new ArrayList<>();
            int num = root.get("num").asInt();
            for (int index = 0; index < num; ++index) {
                this.ids.add(new ObjectID(root.get(String.valueOf(index)).asInt()));
            }
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = false)
    public static class GetBuffersReply extends Reply {
        private List<Payload> payloads;

        public void Put(ObjectNode root, List<Payload> objects) {
            root.put("type", "get_buffers_reply");
            int index = 0;
            for (val payload : objects) {
                root.putPOJO(String.valueOf(index), payload);
            }
            root.put("num", objects.size());
        }

        @Override
        public void Get(JsonNode root) throws VineyardException {
            check(root, "get_buffers_reply");
            this.payloads = new ArrayList<>();
            for (int index = 0; index < root.get("num").asInt(); ++index) {
                val payload = Payload.fromJson(root.get(String.valueOf(index)));
                this.payloads.add(payload);
            }
        }
    }
}
