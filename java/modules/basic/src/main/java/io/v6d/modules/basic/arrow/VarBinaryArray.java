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

import io.v6d.core.client.Context;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import java.util.Arrays;
import lombok.*;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

/** Hello world! */
public class VarBinaryArray extends Array {
    private VarBinaryVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register(
                        "vineyard::VarBinaryArray",
                        new VarBinaryArrayResolver());
    }

    public VarBinaryArray(final ObjectMeta meta, Buffer buffer, Buffer offset, long length) {
        super(meta);
        Context.println("VarBinaryArray: " + meta.toString());
        this.array = new VarBinaryVector("", Arrow.default_allocator);
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0),
                Arrays.asList(null, offset.getBuffer(), buffer.getBuffer()));
        Context.println("length:" + length);
        this.array.setValueCount((int) length);
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
        VarBinaryArray that = (VarBinaryArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class VarBinaryArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        val buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_data_"));
        val offsets_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_offsets_"));
        return new VarBinaryArray(meta, buffer, offsets_buffer, meta.getLongValue("length_"));
    }
}
