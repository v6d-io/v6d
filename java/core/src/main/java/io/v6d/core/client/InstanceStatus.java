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
package io.v6d.core.client;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import io.v6d.core.common.util.InstanceID;
import lombok.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class InstanceStatus {
    @JsonProperty private InstanceID instanceID;
    @JsonProperty private String deployment;
    @JsonProperty private long memory_usage;
    @JsonProperty private long memory_limit;
    @JsonProperty private long deferred_requests;
    @JsonProperty private long ipc_connections;
    @JsonProperty private long rpc_connections;

    public static InstanceStatus fromJson(JsonNode root) {
        val status = new InstanceStatus();
        status.instanceID = new InstanceID(root.get("instance_id").longValue());
        status.deployment = root.get("deployment").textValue();
        status.memory_usage = root.get("memory_usage").longValue();
        status.memory_limit = root.get("memory_limit").longValue();
        status.deferred_requests = root.get("deferred_requests").longValue();
        status.ipc_connections = root.get("ipc_connections").longValue();
        status.rpc_connections = root.get("rpc_connections").longValue();
        return status;
    }
}
