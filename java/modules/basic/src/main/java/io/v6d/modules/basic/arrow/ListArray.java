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
import io.v6d.core.common.util.VineyardException.NotImplemented;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import lombok.val;

import org.apache.arrow.flatbuf.Int;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.ArrowType.ArrowTypeID;

/** Hello world! */
public class ListArray extends Array {
    private ListVector array;

    public static void instantiate() {
        // for(ArrowType.ArrowTypeID type : ArrowType.ArrowTypeID.values()) {
        //     Context.println("register:" + "vineyard::ListArray<" + type.name() + ">");
        //     ObjectFactory.getFactory()
        //             .register(
        //                     "vineyard::ListArray<" + type.name() + ">",
        //                     new ListArrayResolver());
        // }
        ObjectFactory.getFactory()
                .register(
                        "vineyard::ListArray",
                        new ListArrayResolver());
    }

    public ListArray(ObjectMeta meta, List<ArrowBuf> buffer, long length, FieldType type, List<ArrowBuf> childBuf) {
        super(meta);
        this.array = new ListVector("", Arrow.default_allocator, type, null);
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), buffer);
    }

    public List get(int index) {
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
        ListArray that = (ListArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class ListArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        int typeID = meta.getIntValue("valueType_");
        ArrowTypeID type = ArrowTypeID.values()[typeID];
        int bitWidth = meta.getIntValue("bitWidth_");
        int listLength = meta.getIntValue("listLength_");
        switch (type) {
            case Int:
                Context.println("build int list array!");
                List<ArrowBuf> dataBufs = new ArrayList<>();
                List<ArrowBuf> bufs = new ArrayList<>();
                bufs.add(((Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("null_bitmap_"))).getBuffer());
                bufs.add(((Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_offsets_"))).getBuffer());
                for (int i = 0; i < listLength; i++) {
                    dataBufs.add(((Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_" + String.valueOf(i) + "_"))).getBuffer());
                }
                return new ListArray(meta, bufs, listLength, new FieldType(true, new ArrowType.Int(bitWidth, true), null), dataBufs);
            default:
                return null;
        }
    }
}
