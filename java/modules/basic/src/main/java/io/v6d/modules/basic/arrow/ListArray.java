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
import io.v6d.modules.basic.arrow.util.ArrowVectorUtils;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.types.pojo.Field;

/** Hello world! */
public class ListArray extends Array {
    private ListVector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register(
                        "vineyard::ListArray",
                        new ListArrayResolver());
    }

    public ListArray(ObjectMeta meta, Queue<ArrowBuf> bufs, Queue<Integer> valueCountList, Field listVectorField) {
        super(meta);
        Context.println("stage 5");
        this.array = ListVector.empty("", Arrow.default_allocator);
        Context.println("stage 6");

        try {
            ArrowVectorUtils.buildArrowVector(this.array, bufs, valueCountList, listVectorField);
        } catch (Exception e) {
            Context.println("Create list array error! Message:" + e.getMessage());
        }
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
        Queue<ArrowBuf> bufs = new LinkedList<>();
        Queue<Integer> valueCountQueue = new LinkedList<>();

        // bufs
        int bufsNum = meta.getIntValue("bufsNum_");
        int valueCountNum = meta.getIntValue("valueCountNum_");
        for (int i = 0; i < bufsNum; i++) {
            bufs.add(((Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_" + String.valueOf(i) + "_"))).getBuffer());
        }
        for (int i = 0; i < valueCountNum; i++) {
            valueCountQueue.add(meta.getIntValue("valueCount_" + String.valueOf(i) + "_"));
        }

        Schema schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));
        List<Field> fields = schema.getSchema().getFields();
        for (int i = 0; i < fields.size(); i++) {
            ArrowVectorUtils.printFields(fields.get(i));
        }
        return new ListArray(meta, bufs, valueCountQueue, fields.get(0));
    }
}
