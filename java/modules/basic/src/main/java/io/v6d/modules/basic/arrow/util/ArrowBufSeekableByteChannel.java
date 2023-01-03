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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import org.apache.arrow.memory.ArrowBuf;

/** A {@link SeekableByteChannel} backed by a byte array. */
public class ArrowBufSeekableByteChannel implements SeekableByteChannel {
    private ArrowBuf arrowBuf;
    private int position = 0;

    /** Construct a new object using the given byteArray as a backing store. */
    public ArrowBufSeekableByteChannel(ArrowBuf arrowBuf) {
        if (arrowBuf == null) {
            throw new NullPointerException();
        }
        this.arrowBuf = arrowBuf;
    }

    @Override
    public boolean isOpen() {
        return arrowBuf != null;
    }

    @Override
    public void close() throws IOException {
        arrowBuf = null;
    }

    @Override
    public int read(final ByteBuffer dst) throws IOException {
        int remainingInBuf = (int) arrowBuf.capacity() - this.position;
        int length = Math.min(dst.remaining(), remainingInBuf);
        for (int i = this.position; i < length + this.position; ++i) {
            dst.put(arrowBuf.getByte(i));
        }
        this.position += length;
        return length;
    }

    @Override
    public long position() throws IOException {
        return this.position;
    }

    @Override
    public SeekableByteChannel position(final long newPosition) throws IOException {
        this.position = (int) newPosition;
        return this;
    }

    @Override
    public long size() throws IOException {
        return this.arrowBuf.capacity();
    }

    @Override
    public int write(final ByteBuffer src) throws IOException {
        throw new UnsupportedOperationException("Read only");
    }

    @Override
    public SeekableByteChannel truncate(final long size) throws IOException {
        throw new UnsupportedOperationException("Read only");
    }
}
