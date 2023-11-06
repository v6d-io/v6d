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
import java.util.Arrays;
import lombok.val;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

public class Int8ArrayBuilder implements ArrayBuilder {
    private BufferBuilder dataBufferBuilder;
    private BufferBuilder validityBufferBuilder;
    private TinyIntVector array;

    public Int8ArrayBuilder(IPCClient client, long length) throws VineyardException {
        this.array = new TinyIntVector("", Arrow.default_allocator);
        this.dataBufferBuilder =
                new BufferBuilder(client, this.array.getBufferSizeFor((int) length));
        this.array.loadFieldBuffers(
                new ArrowFieldNode(length, 0), Arrays.asList(null, dataBufferBuilder.getBuffer()));
    }

    @Override
    public void build(Client client) throws VineyardException {
        ArrowBuf buf = array.getValidityBuffer();
        validityBufferBuilder = new BufferBuilder((IPCClient) client, buf, buf.capacity());
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::NumericArray<int8>");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", array.getNullCount());
        meta.setValue("offset_", 0);
        meta.addMember("data_buffer_", dataBufferBuilder.seal(client));
        meta.addMember("validity_buffer_", validityBufferBuilder.seal(client));
        return client.createMetaData(meta);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }

    @Override
    public void shrink(Client client, long size) throws VineyardException {
        this.dataBufferBuilder.shrink(client, this.array.getBufferSizeFor((int) size));
        this.array.setValueCount((int) size);
    }

    void set(int index, byte value) {
        this.array.set(index, value);
    }
}
