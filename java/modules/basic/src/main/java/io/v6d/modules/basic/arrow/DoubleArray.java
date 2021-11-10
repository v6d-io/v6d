/** Copyright 2020-2021 Alibaba Group Holding Limited.
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

import static io.v6d.modules.basic.arrow.Arrow.logger;

import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import java.util.Arrays;
import lombok.*;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

/** Hello world! */
public class DoubleArray extends Array {
    private Float8Vector array;

    public static void instantiate() {
        ObjectFactory.getFactory()
                .register("vineyard::NumericArray<double>", new DoubleArrayResolver());
    }

    public DoubleArray(Buffer buffer, long length) {
        this.array = new Float8Vector("", Arrow.default_allocator);
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), Arrays.asList(null, buffer.getBuffer()));
    }

    public double get(int index) {
        return this.array.get(index);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }
}

class DoubleArrayResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(ObjectMeta meta) {
        logger.debug("double array resolver: from metadata {}", meta);
        val buffer = (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        return new DoubleArray(buffer, meta.getLongValue("length_"));
    }
}
