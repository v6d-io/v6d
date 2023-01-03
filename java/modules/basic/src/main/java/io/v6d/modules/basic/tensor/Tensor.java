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
package io.v6d.modules.basic.tensor;

import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.Buffer;
import io.v6d.modules.basic.columnar.ColumnarData;
import java.util.Arrays;
import java.util.Collection;
import lombok.*;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.Types;

public class Tensor extends Object {
    private Buffer buffer;
    private Collection<Integer> shape;
    private Types.MinorType dtype;
    private FieldVector array;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::Tensor<int>", new Int32TensorResolver());
        ObjectFactory.getFactory().register("vineyard::Tensor<int64>", new Int64TensorResolver());
        ObjectFactory.getFactory().register("vineyard::Tensor<float>", new FloatTensorResolver());
        ObjectFactory.getFactory().register("vineyard::Tensor<double>", new DoubleTensorResolver());
    }

    public Tensor(
            final ObjectMeta meta,
            Buffer buffer,
            Collection<Integer> shape,
            Types.MinorType dtype,
            FieldVector array) {
        super(meta);

        this.buffer = buffer;
        this.shape = shape;
        this.dtype = dtype;
        this.array = array;

        var length = 1;
        if (shape.isEmpty()) {
            length = 0;
        } else {
            for (val item : shape) {
                length *= item;
            }
        }
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), Arrays.asList(null, buffer.getBuffer()));
    }

    public FieldVector getArray() {
        return array;
    }

    public ColumnarData columnar() {
        return new ColumnarData(getArray());
    }
}

abstract class TensorResolver extends ObjectFactory.Resolver {
    @Override
    public Object resolve(final ObjectMeta meta) {
        val buffer = (Buffer) meta.getMember("buffer_");
        val shape = meta.<Integer>getListValue("shape_");
        return new Tensor(meta, buffer, shape, dtype(), array());
    }

    protected abstract Types.MinorType dtype();

    protected abstract FieldVector array();
}

class Int32TensorResolver extends TensorResolver {
    @Override
    protected Types.MinorType dtype() {
        return Types.MinorType.INT;
    }

    @Override
    protected FieldVector array() {
        return new IntVector("", Arrow.default_allocator);
    }
}

class Int64TensorResolver extends TensorResolver {
    @Override
    protected Types.MinorType dtype() {
        return Types.MinorType.BIGINT;
    }

    @Override
    protected FieldVector array() {
        return new BigIntVector("", Arrow.default_allocator);
    }
}

class FloatTensorResolver extends TensorResolver {
    @Override
    protected Types.MinorType dtype() {
        return Types.MinorType.FLOAT4;
    }

    @Override
    protected FieldVector array() {
        return new Float4Vector("", Arrow.default_allocator);
    }
}

class DoubleTensorResolver extends TensorResolver {
    @Override
    protected Types.MinorType dtype() {
        return Types.MinorType.FLOAT8;
    }

    @Override
    protected FieldVector array() {
        return new Float8Vector("", Arrow.default_allocator);
    }
}
