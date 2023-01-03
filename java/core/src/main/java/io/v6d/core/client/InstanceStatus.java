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
package io.v6d.core.client;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import io.v6d.core.common.util.InstanceID;
import io.v6d.core.common.util.JSON;
import java.io.Serializable;
import lombok.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class InstanceStatus implements Serializable {
    @JsonProperty private InstanceID instanceID;
    @JsonProperty private String deployment;
    @JsonProperty private long memory_usage;
    @JsonProperty private long memory_limit;
    @JsonProperty private long deferred_requests;
    @JsonProperty private long ipc_connections;
    @JsonProperty private long rpc_connections;

    @JsonProperty private InstanceID hostid;
    @JsonProperty private String hostname;
    @JsonProperty private String nodename;

    public static InstanceStatus fromJson(JsonNode root) {
        val status = new InstanceStatus();
        status.instanceID = new InstanceID(JSON.getLongMaybe(root, "instance_id", -1));
        status.deployment = JSON.getTextMaybe(root, "deployment", "unknown");
        status.memory_usage = JSON.getLongMaybe(root, "memory_usage", -1);
        status.memory_limit = JSON.getLongMaybe(root, "memory_limit", -1);
        status.deferred_requests = JSON.getLongMaybe(root, "deferred_requests", -1);
        status.ipc_connections = JSON.getLongMaybe(root, "ipc_connections", -1);
        status.rpc_connections = JSON.getLongMaybe(root, "rpc_connections", -1);

        status.hostid = new InstanceID(JSON.getLongMaybe(root, "hostid", -1));
        status.hostname = JSON.getTextMaybe(root, "hostname", "localhost");
        status.nodename = JSON.getTextMaybe(root, "nodename", "localhost");
        return status;
    }
}
