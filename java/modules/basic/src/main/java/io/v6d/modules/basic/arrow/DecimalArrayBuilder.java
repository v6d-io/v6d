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

import io.v6d.core.client.Client;
import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import java.util.Arrays;
import lombok.val;

import org.apache.arrow.vector.BaseFixedWidthVector;
import org.apache.arrow.vector.Decimal256Vector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType.Decimal;

public class DecimalArrayBuilder implements ArrayBuilder {
    private BufferBuilder buffer;
    private BaseFixedWidthVector array;
    int maxPrecision;
    int maxScale;
    int bitWidth;

    public DecimalArrayBuilder(IPCClient client, long length, int maxPrecision, int maxScale, int bitWidth) throws VineyardException {
        this.maxPrecision = maxPrecision;
        this.maxScale = maxScale;
        this.bitWidth = bitWidth;
        if (bitWidth == 128) {
            this.array = new DecimalVector("", Arrow.default_allocator, maxPrecision, maxScale);
        } else {
            this.array = new Decimal256Vector("", Arrow.default_allocator, maxPrecision, maxScale);
        }
        this.buffer = new BufferBuilder(client, this.array.getBufferSizeFor((int) length));
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), Arrays.asList(null, buffer.getBuffer()));
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::DecimalArray<" + String.valueOf(bitWidth) + ">");
        Context.println("vineyard::DecimalArray<" + String.valueOf(bitWidth) + ">");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", 0);
        meta.setValue("offset_", 0);
        meta.addMember("buffer_", buffer.seal(client));
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
        meta.setValue("maxPrecision_", maxPrecision);
        meta.setValue("maxScale_", maxScale);
        meta.setValue("bitWidth_", bitWidth);
        return client.createMetaData(meta);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }

    @Override
    public void shrink(Client client, long size) throws VineyardException {
        this.buffer.shrink(client, this.array.getBufferSizeFor((int) size));
        this.array.setValueCount((int) size);
    }
}
