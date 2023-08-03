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
package io.v6d.modules.basic.arrow;

import static com.google.common.base.MoreObjects.toStringHelper;

import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Objects;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import lombok.val;
// import org.apache.arrow.memory.ArrowBuf;
// import org.apache.arrow.memory.ReferenceManager;
import io.netty.buffer.ArrowBuf;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.LargeBuffer;
import io.netty.buffer.PooledByteBufAllocatorL;
import io.netty.buffer.UnsafeDirectLittleEndian;
import io.netty.util.internal.PlatformDependent;

public class Buffer extends Object {
    private ArrowBuf buffer;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::Blob", new BufferResolver());
    }

    public Buffer(final ObjectMeta metadata, long address, long length) {
        super(metadata);
        // this.buffer = new ArrowBuf(ReferenceManager.NO_OP, null /* not needed */, length, address);

        // PooledByteBufAllocatorL allocator = new PooledByteBufAllocatorL();
        // UnsafeDirectLittleEndian unsafeDirectLittleEndian = allocator.allocate((int)length);
        buffer = Arrow.default_allocator.buffer((int)length);

        System.out.println("read from vineyard");
        PlatformDependent.copyMemory(address, buffer.memoryAddress(), length);
        for (int i = 0; i < length; i++) {
            System.out.println("data:" + buffer.getByte(i));
        }
        // buffer = new ArrowBuf(
        //          new AtomicInteger(), null, unsafeDirectLittleEndian, null, null, 0, (int) length, false);
    }

    public ArrowBuf getBuffer() {
        return buffer;
    }

    public long length() {
        return this.buffer.capacity();
    }

    @Override
    public boolean equals(java.lang.Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Buffer buffer1 = (Buffer) o;
        return Objects.equal(buffer, buffer1.buffer);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(buffer);
    }

    @Override
    public String toString() {
        return toStringHelper(this)
                .add("id", meta.getId())
                .add("length", buffer.capacity())
                .toString();
    }
}

class BufferResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta metadata) {
        System.out.println("resolve");
        val buffer = metadata.getBuffer(metadata.getId());
        // System.out.println("=============");
        // for (int i = 0; i < buffer.getSize() ; i++) {
        //     System.out.println("data:" + (int)(buffer.toString().getBytes()[i]));
        // }
        return new Buffer(metadata, buffer.getPointer(), buffer.getSize());
    }
}
