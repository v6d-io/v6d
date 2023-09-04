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
import io.v6d.core.client.ds.Buffer;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import lombok.*;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.ReferenceManager;
import org.apache.arrow.memory.util.MemoryUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BufferBuilder implements ObjectBuilder {
    private Logger logger = LoggerFactory.getLogger(BufferBuilder.class);

    private final Buffer buffer;
    private final ArrowBuf arrowBuf;

    public BufferBuilder(IPCClient client, long size) throws VineyardException {
        this.buffer = client.createBuffer(size);
        if (size == 0) {
            this.arrowBuf = null;
        } else {
            this.arrowBuf =
                    new ArrowBuf(
                            ReferenceManager.NO_OP, null, buffer.getSize(), buffer.getPointer());
        }
    }

    public BufferBuilder(IPCClient client, final ArrowBuf buffer) throws VineyardException {
        this(client, buffer, buffer.capacity());
        logger.warn(
                "Construct buffer builder without explicit size is dangerous, will over commit the memory");
    }

    public BufferBuilder(IPCClient client, final ArrowBuf buffer, final long capacity)
            throws VineyardException {
        this.buffer = client.createBuffer(capacity);
        this.arrowBuf = buffer;
        if (capacity > 0) {
            MemoryUtil.UNSAFE.copyMemory(
                    this.arrowBuf.memoryAddress(), this.buffer.getPointer(), capacity);
        }
    }

    public static BufferBuilder fromByteArray(IPCClient client, byte[] bytes)
            throws VineyardException {
        val builder = new BufferBuilder(client, bytes.length);
        if (bytes.length > 0) {
            builder.arrowBuf.writeBytes(bytes);
        }
        return builder;
    }

    @SneakyThrows(VineyardException.class)
    public static ObjectMeta empty(Client client) {
        val meta = ObjectMeta.empty();
        meta.setId(ObjectID.EmptyBlobID); // blob's builder is a special case
        meta.setInstanceId(client.getInstanceId());

        meta.setTypename("vineyard::Blob");
        meta.setNBytes(0);
        meta.setValue("length", 0);

        // to make resolving the returned object metadata possible
        meta.setBufferUnchecked(ObjectID.EmptyBlobID, new Buffer(ObjectID.EmptyBlobID, 0, 0));

        return meta; // n.b.: blob: no create meta action
    }

    public void shrink(Client client, long size) throws VineyardException {
        client.shrinkBuffer(this.buffer.getObjectId(), size);
        this.buffer.setSize(size);
    }

    @Override
    public void build(Client client) throws VineyardException {
        // TODO: re-mmap as readonly
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        client.sealBuffer(buffer.getObjectId());
        val meta = ObjectMeta.empty();
        meta.setId(buffer.getObjectId()); // blob's builder is a special case
        meta.setInstanceId(client.getInstanceId());

        meta.setTypename("vineyard::Blob");
        meta.setNBytes(buffer.getSize());
        meta.setValue("length", buffer.getSize());

        // to make resolving the returned object metadata possible
        meta.setBufferUnchecked(buffer.getObjectId(), buffer);

        client.sealBuffer(buffer.getObjectId());
        return meta; // n.b.: blob: no create meta action
    }

    public ArrowBuf getBuffer() {
        return arrowBuf;
    }

    public long length() {
        return this.buffer.getSize();
    }
}
