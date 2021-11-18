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
import io.v6d.core.client.ds.Buffer;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import lombok.*;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.ReferenceManager;

public class BufferBuilder implements ObjectBuilder {
    private final Buffer buffer;
    private final ArrowBuf arrowBuf;

    public BufferBuilder(IPCClient client, long size) throws VineyardException {
        this.buffer = client.createBuffer(size);
        this.arrowBuf =
                new ArrowBuf(ReferenceManager.NO_OP, null, buffer.getSize(), buffer.getPointer());
    }

    public static BufferBuilder fromByteArray(IPCClient client, byte[] bytes)
            throws VineyardException {
        val builder = new BufferBuilder(client, bytes.length);
        builder.arrowBuf.writeBytes(bytes);
        return builder;
    }

    @Override
    public void build(Client client) throws VineyardException {
        // TODO: re-mmap as readonly
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        meta.setId(buffer.getObjectId()); // blob's builder is a special case
        meta.setInstanceId(client.getInstanceId());

        meta.setTypename("vineyard::Blob");
        meta.setNBytes(buffer.getSize());
        meta.setValue("length", buffer.getSize());

        // to make resolving the returned object metadata possible
        meta.setBufferUnchecked(buffer.getObjectId(), buffer);

        return meta; // n.b.: blob: no create meta action
    }

    public ArrowBuf getBuffer() {
        return arrowBuf;
    }

    public long length() {
        return this.buffer.getSize();
    }
}
