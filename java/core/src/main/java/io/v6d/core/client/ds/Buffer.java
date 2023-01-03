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
package io.v6d.core.client.ds;

import io.v6d.core.common.util.ObjectID;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
public class Buffer {
    private ObjectID objectId;
    private long pointer;
    private long size;

    public Buffer() {
        this.objectId = ObjectID.InvalidObjectID;
        this.pointer = 0;
        this.size = 0;
    }

    public Buffer(ObjectID objectId, long pointer, long size) {
        this.objectId = objectId;
        this.pointer = pointer;
        this.size = size;
    }

    public static Buffer empty() {
        return new Buffer(ObjectID.EmptyBlobID, 0, 0);
    }

    public boolean isNull() {
        return this.size == 0;
    }
}
