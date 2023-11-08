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
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.NullVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

public class NullArray extends Array {
    private NullVector array;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::NullArray", new NUllArrayResolver());
    }

    public NullArray(final ObjectMeta meta, long length) {
        super(meta);
        this.array = new NullVector();
        this.array.loadFieldBuffers(new ArrowFieldNode(length, 0), Arrays.asList());
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
        return this.length() == ((NullArray) o).length();
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(array);
    }
}

class NUllArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        return new NullArray(meta, meta.getLongValue("length_"));
    }
}
