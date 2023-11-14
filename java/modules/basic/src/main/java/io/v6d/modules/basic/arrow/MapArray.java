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
import io.v6d.modules.basic.arrow.util.ArrowVectorUtils;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.types.pojo.Field;

public class MapArray extends Array {
    private MapVector array;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::MapArray", new MapArrayResolver());
    }

    public MapArray(
            ObjectMeta meta,
            Queue<ArrowBuf> bufs,
            Queue<Integer> valueCountList,
            Field structVectorField) {
        super(meta);
        this.array = MapVector.empty("", Arrow.default_allocator, true);
        ArrowVectorUtils.buildArrowVector(this.array, bufs, valueCountList, structVectorField);
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
        MapArray that = (MapArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class MapArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        Queue<ArrowBuf> bufs = new LinkedList<>();
        Queue<Integer> valueCountQueue = new LinkedList<>();

        int bufsNum = meta.getIntValue("bufs_num_");
        int valueCountNum = meta.getIntValue("value_count_num_");
        for (int i = 0; i < bufsNum; i++) {
            ObjectMeta bufMeta = meta.getMemberMeta("buffer_" + String.valueOf(i) + "_");
            bufs.add(((Buffer) ObjectFactory.getFactory().resolve(bufMeta)).getBuffer());
        }
        for (int i = 0; i < valueCountNum; i++) {
            valueCountQueue.add(meta.getIntValue("value_count_" + String.valueOf(i) + "_"));
        }

        Schema schema = (Schema) new SchemaResolver().resolve(meta.getMemberMeta("schema_"));
        List<Field> fields = schema.getSchema().getFields();
        return new MapArray(meta, bufs, valueCountQueue, fields.get(0));
    }
}
