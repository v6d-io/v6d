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
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.TimeStampNanoVector;
import org.apache.arrow.vector.TimeStampSecVector;
import org.apache.arrow.vector.TimeStampVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.TimeUnit;

public class TimestampArray extends Array {
    private TimeStampVector array;
    private TimeUnit timeUnit;

    public static void instantiate() {
        for (TimeUnit timeUnit : TimeUnit.values()) {
            ObjectFactory.getFactory()
                    .register(
                            "vineyard::Timestamp<" + timeUnit.toString() + ">",
                            new TimestampArrayResolver());
        }
    }

    public TimestampArray(
            ObjectMeta meta, List<ArrowBuf> buffers, long length, int nullCount, short timeUnitID) {
        super(meta);
        switch (TimeUnit.values()[timeUnitID]) {
            case MICROSECOND:
                this.array = new TimeStampMicroVector("", Arrow.default_allocator);
                break;
            case NANOSECOND:
                this.array = new TimeStampNanoVector("", Arrow.default_allocator);
                break;
            case SECOND:
                this.array = new TimeStampSecVector("", Arrow.default_allocator);
                break;
            case MILLISECOND:
                this.array = new TimeStampMilliVector("", Arrow.default_allocator);
                break;
        }
        this.array.loadFieldBuffers(new ArrowFieldNode(length, nullCount), buffers);
        this.timeUnit = TimeUnit.values()[timeUnitID];
    }

    public long get(int index) {
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
        TimestampArray that = (TimestampArray) o;
        return Objects.equal(array, that.array);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class TimestampArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        Buffer data_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        Buffer validity_buffer =
                (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("null_bitmap_"));
        int null_count = meta.getIntValue("null_count_");
        int length = meta.getIntValue("length_");
        return new TimestampArray(
                meta,
                Arrays.asList(validity_buffer.getBuffer(), data_buffer.getBuffer()),
                length,
                null_count,
                (short) meta.getLongValue("time_unit_id_"));
    }
}
