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

import com.google.common.base.Objects;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import java.util.Arrays;
import java.util.List;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

public class Int32Array extends Array {
    private IntVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register("vineyard::NumericArray<int>", new Int32ArrayResolver());
        ObjectFactory.getFactory()
                .register("vineyard::NumericArray<uint>", new Int32ArrayResolver());
    }

    public Int32Array(ObjectMeta meta, List<ArrowBuf> buffers, long length, int nullCount) {
        super(meta);
        this.array = new IntVector("", Arrow.default_allocator);
        this.array.loadFieldBuffers(new ArrowFieldNode(length, nullCount), buffers);
    }

    public double get(int index) {
        return this.array.get(index);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }

    @Override
    public boolean equals(java.lang.Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Int32Array that = (Int32Array) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class Int32ArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        Buffer dataBuffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        Buffer validityBuffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("null_bitmap_"));
        int nullCount = meta.getIntValue("null_count_");
        return new Int32Array(
                meta,
                Arrays.asList(validityBuffer.getBuffer(), dataBuffer.getBuffer()),
                meta.getLongValue("length_"),
                nullCount);
    }
}
