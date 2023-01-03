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

import java.io.Serializable;
import lombok.*;
import lombok.EqualsAndHashCode;

/** Vineyard ObjectID definition. */
@EqualsAndHashCode(callSuper = false)
public class ObjectID implements Comparable<ObjectID>, Serializable {
    private long id = -1L;

    public static ObjectID InvalidObjectID = new ObjectID(-1L);

    public static ObjectID EmptyBlobID = new ObjectID(0x8000000000000000L);

    public ObjectID(long id) {
        this.id = id;
    }

    public static ObjectID fromString(String id) {
        return new ObjectID(Long.parseUnsignedLong(id.substring(1), 16));
    }

    public long value() {
        return this.id;
    }

    public boolean isBlob() {
        return (this.id & 0x8000000000000000L) != 0L;
    }

    @Override
    public String toString() {
        return String.format("o%016x", id);
    }

    @Override
    public int compareTo(@NonNull ObjectID other) {
        return (int) (this.id - other.id);
    }
}
