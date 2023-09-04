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
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.NullVector;

public class NullArrayBuilder implements ArrayBuilder {
    private final NullVector array;

    public NullArrayBuilder(IPCClient client, final NullVector vector) throws VineyardException {
        this.array = vector;
    }

    public NullArrayBuilder(IPCClient client, long length) throws VineyardException {
        this.array = new NullVector();
        this.array.setValueCount((int) length);
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setTypename("vineyard::NullArray");
        meta.setNBytes(0);
        meta.setValue("length_", array.getValueCount());
        meta.setValue("null_count_", 0);
        meta.setValue("offset_", 0);
        meta.addMember("buffer_data_", BufferBuilder.empty(client));
        meta.addMember("buffer_offsets_", BufferBuilder.empty(client));
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
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
