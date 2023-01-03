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
package io.v6d.core.common.memory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import io.v6d.core.common.util.ObjectID;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.val;

@Data
@EqualsAndHashCode(callSuper = false)
public class Payload {
    @JsonProperty private ObjectID objectID;
    @JsonProperty private int storeFD;
    @JsonProperty private int arenaFD;
    @JsonProperty private long dataOffset;
    @JsonProperty private long dataSize;
    @JsonProperty private long mapSize;
    @JsonProperty private long pointer; // uint8_t *

    private Payload() {
        this.objectID = ObjectID.EmptyBlobID;
        this.storeFD = -1;
        this.arenaFD = -1;
        this.dataOffset = 0;
        this.dataSize = 0;
        this.mapSize = 0;
        this.pointer = 0;
    }

    public static Payload makeEmpty() {
        // FIXME use a const static payload object.
        return new Payload();
    }

    public Payload(
            ObjectID objectID,
            int storeFD,
            int arenaFD,
            long dataOffset,
            long dataSize,
            long mapSize,
            long pointer) {
        this.objectID = objectID;
        this.storeFD = storeFD;
        this.arenaFD = arenaFD;
        this.dataOffset = dataOffset;
        this.dataSize = dataSize;
        this.mapSize = mapSize;
        this.pointer = pointer;
    }

    public Payload(
            ObjectID objectID,
            int storeFD,
            long dataOffset,
            long dataSize,
            long mapSize,
            long pointer) {
        this.objectID = objectID;
        this.storeFD = storeFD;
        this.arenaFD = -1;
        this.dataOffset = dataOffset;
        this.dataSize = dataSize;
        this.mapSize = mapSize;
        this.pointer = pointer;
    }

    public static Payload fromJson(JsonNode root) {
        val payload = new Payload();
        payload.objectID = new ObjectID(root.get("object_id").longValue());
        payload.storeFD = root.get("store_fd").intValue();
        payload.dataOffset = root.get("data_offset").longValue();
        payload.dataSize = root.get("data_size").longValue();
        payload.mapSize = root.get("map_size").longValue();
        return payload;
    }
}
