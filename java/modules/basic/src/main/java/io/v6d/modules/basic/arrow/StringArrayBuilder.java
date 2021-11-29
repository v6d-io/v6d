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

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import lombok.*;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.LargeVarCharVector;
import org.apache.arrow.vector.util.Text;

public class StringArrayBuilder implements ArrayBuilder {
    private LargeVarCharVector array;

    private BufferBuilder data_buffer_builder;
    private BufferBuilder offset_buffer_builder;

    public StringArrayBuilder(IPCClient client, final LargeVarCharVector vector)
            throws VineyardException {
        this.array = vector;
    }

    public StringArrayBuilder(IPCClient client, long length) throws VineyardException {
        this.array = new LargeVarCharVector("", Arrow.default_allocator);
        this.array.setValueCount((int) length);
    }

    @Override
    public void build(Client client) throws VineyardException {
        // FIXME the builder is a IPCClient, but RPCClient should be able work as well
        this.data_buffer_builder =
                new BufferBuilder((IPCClient) client, this.array.getDataBuffer());
        this.offset_buffer_builder =
                new BufferBuilder((IPCClient) client, this.array.getOffsetBuffer());
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::BaseBinaryArray<arrow::LargeStringArray>");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", 0);
        meta.setValue("offset_", 0);
        meta.addMember("buffer_data_", data_buffer_builder.seal(client));
        meta.addMember("buffer_offsets_", offset_buffer_builder.seal(client));
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
        return client.createMetaData(meta);
    }

    void set(int index, String value) {
        this.array.setSafe(index, value.getBytes());
    }

    void set(int index, Text value) {
        this.array.set(index, value);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }
}
