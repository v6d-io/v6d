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
import io.v6d.core.common.util.VineyardException;
import java.util.*;
import lombok.val;

public class BufferSet {
    private Set<ObjectID> buffer_ids;
    private Map<ObjectID, Buffer> buffers;

    public BufferSet() {
        this.buffer_ids = new HashSet<>();
        this.buffers = new HashMap<>();
    }

    public Set<ObjectID> allBufferIds() {
        return this.buffer_ids;
    }

    public Map<ObjectID, Buffer> allBuffers() {
        return this.buffers;
    }

    public void emplace(ObjectID id) throws VineyardException {
        if (this.buffers.getOrDefault(id, null) != null) {
            throw new VineyardException.Invalid(
                    "Invalid internal state: the buffer shouldn't has been filled, id = " + id);
        }
        buffer_ids.add(id);
        buffers.put(id, null);
    }

    public void emplace(ObjectID id, Buffer buffer) throws VineyardException {
        if (!this.buffers.containsKey(id)) {
            throw new VineyardException.Invalid(
                    "Invalid internal state: no such buffer defined, id = " + id);
        }
        if (this.buffers.get(id) != null) {
            throw new VineyardException.Invalid(
                    "Invalid internal state: duplicated buffer, id = " + id);
        }
        this.buffers.put(id, buffer);
    }

    public void emplaceUnchecked(ObjectID id, Buffer buffer) throws VineyardException {
        this.buffer_ids.add(id);
        this.buffers.put(id, buffer);
    }

    public void extend(BufferSet other) {
        for (val item : other.buffers.entrySet()) {
            buffers.put(item.getKey(), item.getValue());
        }
    }

    public boolean contains(ObjectID id) {
        return this.buffers.containsKey(id);
    }

    public Buffer get(ObjectID id) {
        return this.buffers.getOrDefault(id, null);
    }

    @Override
    public String toString() {
        return "BufferSet{" + "buffer_ids=" + buffer_ids + ", buffers=" + buffers + '}';
    }
}
