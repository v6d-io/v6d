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

import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;

public class DateArrayBuilder implements ArrayBuilder {
    private BufferBuilder buffer;
    private DateDayVector array;

    public DateArrayBuilder(IPCClient client, long length) throws VineyardException {
        this.array = new DateDayVector("", Arrow.default_allocator);
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
        meta.setTypename("vineyard::Date<Day>");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", 0);
        meta.setValue("offset_", 0);
        meta.addMember("buffer_", buffer.seal(client));
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
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

    void set(int index, int value) {
        this.array.set(index, value);
    }
}
