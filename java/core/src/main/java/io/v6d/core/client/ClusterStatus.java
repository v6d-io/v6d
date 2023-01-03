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

import com.fasterxml.jackson.databind.JsonNode;
import io.v6d.core.common.util.InstanceID;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ClusterStatus
        implements Serializable, Iterable<Map.Entry<InstanceID, InstanceStatus>> {
    private static final Logger logger = LoggerFactory.getLogger(ClusterStatus.class);

    private final Map<InstanceID, InstanceStatus> instances;

    public ClusterStatus() {
        this.instances = new HashMap<>();
    }

    public boolean hasInstance(InstanceID instanceID) {
        return this.instances.containsKey(instanceID);
    }

    public InstanceStatus getInstance(InstanceID instanceID) {
        return this.instances.get(instanceID);
    }

    public Map<InstanceID, InstanceStatus> getInstances() {
        return this.instances;
    }

    public static ClusterStatus fromJson(JsonNode root) {
        val status = new ClusterStatus();
        root.fields()
                .forEachRemaining(
                        entry -> {
                            val instanceID = InstanceID.fromString(entry.getKey());
                            val instanceStatus = InstanceStatus.fromJson(entry.getValue());
                            status.instances.put(instanceID, instanceStatus);
                        });
        return status;
    }

    @Override
    public Iterator<Map.Entry<InstanceID, InstanceStatus>> iterator() {
        return instances.entrySet().iterator();
    }
}
