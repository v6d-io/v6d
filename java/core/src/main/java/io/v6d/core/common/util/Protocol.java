/** Copyright 2020-2021 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package io.v6d.core.common.util;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Data;
import lombok.EqualsAndHashCode;

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
}
