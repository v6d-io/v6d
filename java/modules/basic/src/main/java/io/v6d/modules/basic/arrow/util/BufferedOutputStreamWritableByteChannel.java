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

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BufferedOutputStreamWritableByteChannel implements WritableByteChannel {
    private static final Logger logger =
            LoggerFactory.getLogger(BufferedOutputStreamWritableByteChannel.class);

    private final BufferedOutputStream output;

    public BufferedOutputStreamWritableByteChannel(BufferedOutputStream output) {
        this.output = output;
    }

    @Override
    public int write(ByteBuffer src) throws IOException {
        int written = 0;
        while (src.hasRemaining()) {
            output.write(src.get());
            written++;
        }
        return written;
    }

    @Override
    public boolean isOpen() {
        return true;
    }

    @Override
    public void close() throws IOException {
        this.output.flush();
    }
}
