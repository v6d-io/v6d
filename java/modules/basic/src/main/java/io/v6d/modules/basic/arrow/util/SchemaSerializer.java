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
package io.v6d.modules.basic.arrow.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ReadChannel;
import org.apache.arrow.vector.ipc.WriteChannel;
import org.apache.arrow.vector.ipc.message.MessageChannelReader;
import org.apache.arrow.vector.ipc.message.MessageResult;
import org.apache.arrow.vector.ipc.message.MessageSerializer;
import org.apache.arrow.vector.types.pojo.Schema;

public class SchemaSerializer {
    /** Deserialize Arrow schema from byte array. */
    public static Schema deserialize(final ArrowBuf buffer, final BufferAllocator allocator)
            throws IOException {
        try (MessageChannelReader schemaReader =
                new MessageChannelReader(
                        new ReadChannel(new ArrowBufSeekableByteChannel(buffer)), allocator)) {

            MessageResult result = schemaReader.readNext();
            if (result == null) {
                throw new IOException("Unexpected end of input. Missing schema.");
            }
            return MessageSerializer.deserializeSchema(result.getMessage());
        }
    }

    public static Schema deserialize(final byte[] buffer, final BufferAllocator allocator)
            throws IOException {
        try (MessageChannelReader schemaReader =
                new MessageChannelReader(
                        new ReadChannel(Channels.newChannel(new ByteArrayInputStream(buffer))),
                        allocator)) {
            MessageResult result = schemaReader.readNext();
            if (result == null) {
                throw new IOException("Unexpected end of input. Missing schema.");
            }
            return MessageSerializer.deserializeSchema(result.getMessage());
        }
    }

    public static byte[] serialize(Schema schema) throws IOException {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        MessageSerializer.serialize(new WriteChannel(Channels.newChannel(out)), schema);
        return out.toByteArray();
    }
}
