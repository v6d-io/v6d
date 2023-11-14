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
import org.apache.arrow.vector.BaseFixedWidthVector;
import org.apache.arrow.vector.Decimal256Vector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

public class DecimalArray extends Array {
    private BaseFixedWidthVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register("vineyard::DecimalArray<128>", new DecimalArrayResolver());
        ObjectFactory.getFactory()
                .register("vineyard::DecimalArray<256>", new DecimalArrayResolver());
    }

    public DecimalArray(
            ObjectMeta meta,
            List<ArrowBuf> buffers,
            int nullCount,
            long length,
            int maxPrecision,
            int maxScale,
            int bitWidth) {
        super(meta);
        if (bitWidth == 128) {
            this.array = new DecimalVector("", Arrow.default_allocator, maxPrecision, maxScale);
        } else {
            this.array = new Decimal256Vector("", Arrow.default_allocator, maxPrecision, maxScale);
        }
        this.array.loadFieldBuffers(new ArrowFieldNode(length, nullCount), buffers);
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
        DecimalArray that = (DecimalArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class DecimalArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        Buffer buffer = (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        Buffer validityBuffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("null_bitmap_"));
        int nullCount = meta.getIntValue("null_count_");
        return new DecimalArray(
                meta,
                Arrays.asList(validityBuffer.getBuffer(), buffer.getBuffer()),
                nullCount,
                meta.getLongValue("length_"),
                meta.getIntValue("max_precision_"),
                meta.getIntValue("max_scale_"),
                meta.getIntValue("bit_width_"));
    }
}
