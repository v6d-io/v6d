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
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import lombok.*;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarBinaryVector;

public class VarBinaryArrayBuilder implements ArrayBuilder {
    private VarBinaryVector array;

    private BufferBuilder dataBufferBuilder;
    private BufferBuilder offsetBufferBuilder;
    private BufferBuilder validityBufferBuilder;

    public VarBinaryArrayBuilder(IPCClient client, long length) throws VineyardException {
        this.array = new VarBinaryVector("", Arrow.default_allocator);
        this.array.setValueCount((int) length);
    }

    @Override
    public void build(Client client) throws VineyardException {
        val offset_buffer_size = (this.array.getValueCount() + 1) * VarBinaryVector.OFFSET_WIDTH;
        val offset_buffer = this.array.getOffsetBuffer();

        val data_buffer_size =
                offset_buffer.getLong(
                        ((long) this.array.getValueCount()) * VarBinaryVector.OFFSET_WIDTH);
        val data_buffer = this.array.getDataBuffer();

        ArrowBuf validityBuffer = array.getValidityBuffer();
        validityBufferBuilder =
                new BufferBuilder((IPCClient) client, validityBuffer, validityBuffer.capacity());

        this.dataBufferBuilder =
                new BufferBuilder((IPCClient) client, data_buffer, data_buffer_size);
        this.offsetBufferBuilder =
                new BufferBuilder((IPCClient) client, offset_buffer, offset_buffer_size);
        this.validityBufferBuilder =
                new BufferBuilder((IPCClient) client, validityBuffer, validityBuffer.capacity());
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::VarBinaryArray");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", array.getNullCount());
        meta.setValue("offset_", 0);
        meta.addMember("buffer_", dataBufferBuilder.seal(client));
        meta.addMember("buffer_offsets_", offsetBufferBuilder.seal(client));
        meta.addMember("null_bitmap_", validityBufferBuilder.seal(client));
        return client.createMetaData(meta);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }

    @Override
    public void shrink(Client client, long size) throws VineyardException {
        this.array.setValueCount((int) size);
    }
}
