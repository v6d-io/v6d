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
import lombok.*;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.util.Text;

public class StringArray extends Array {
    private VarCharVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register(
                        "vineyard::BaseBinaryArray<arrow::StringArray>", new StringArrayResolver());
        ObjectFactory.getFactory().register("vineyard::StringArray", new StringArrayResolver());
    }

    public StringArray(final ObjectMeta meta, List<ArrowBuf> buffers, long length, int nullCount) {
        super(meta);
        this.array = new VarCharVector("", Arrow.default_allocator);
        this.array.loadFieldBuffers(new ArrowFieldNode(length, nullCount), buffers);
    }

    public byte[] get(int index) {
        return this.array.get(index);
    }

    public Text getObject(int index) {
        return this.array.getObject(index);
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
        StringArray that = (StringArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class StringArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        Buffer data_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        Buffer offsets_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_offsets_"));
        Buffer validity_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("null_bitmap_"));
        int null_count = meta.getIntValue("null_count_");
        int length = meta.getIntValue("length_");
        return new StringArray(
                meta,
                Arrays.asList(
                        validity_buffer.getBuffer(),
                        offsets_buffer.getBuffer(),
                        data_buffer.getBuffer()),
                length,
                null_count);
    }
}
