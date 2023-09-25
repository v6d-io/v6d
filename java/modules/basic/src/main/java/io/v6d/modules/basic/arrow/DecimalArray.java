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
import lombok.val;

import org.apache.arrow.vector.BaseFixedWidthVector;
import org.apache.arrow.vector.Decimal256Vector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

/** Hello world! */
public class DecimalArray extends Array {
    private BaseFixedWidthVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register("vineyard::DecimalArray<128>", new DecimalArrayResolver());
        ObjectFactory.getFactory()
                .register("vineyard::DecimalArray<256>", new DecimalArrayResolver());
    }

    public DecimalArray(ObjectMeta meta, Buffer buffer, long length, int maxPrecision, int maxScale, int bitWidth) {
        super(meta);
        if (bitWidth == 128) {
            this.array = new DecimalVector("", Arrow.default_allocator, maxPrecision, maxScale);
        } else {
            this.array = new Decimal256Vector("", Arrow.default_allocator, maxPrecision, maxScale);
        }
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), Arrays.asList(null, buffer.getBuffer()));
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
        val buffer = (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        return new DecimalArray(meta, buffer, meta.getLongValue("length_"), meta.getIntValue("maxPrecision_"), meta.getIntValue("maxScale_"), meta.getIntValue("bitWidth_"));
    }
}
