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
package io.v6d.modules.basic.arrow;

import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import lombok.val;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.ReferenceManager;

public class Buffer extends Object {
    private ArrowBuf buffer;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::Blob", new BufferResolver());
    }

    public Buffer(final ObjectMeta metadata, long address, long length) {
        super(metadata);
        this.buffer = new ArrowBuf(ReferenceManager.NO_OP, null /* not needed */, length, address);
    }

    public ArrowBuf getBuffer() {
        return buffer;
    }

    public long length() {
        return this.buffer.capacity();
    }
}

class BufferResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta metadata) {
        val buffer = metadata.getBuffer(metadata.getId());
        return new Buffer(metadata, buffer.getPointer(), buffer.getSize());
    }
}
