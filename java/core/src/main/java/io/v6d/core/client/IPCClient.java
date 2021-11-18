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
package io.v6d.core.client;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.google.common.io.LittleEndianDataInputStream;
import com.google.common.io.LittleEndianDataOutputStream;
import io.v6d.core.client.ds.Buffer;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.memory.ffi.Fling;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.Protocol.*;
import io.v6d.core.common.util.VineyardException;
import java.io.*;
import java.nio.channels.Channels;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;
import lombok.SneakyThrows;
import lombok.val;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Vineyard IPC client. */
public class IPCClient extends Client {
    private static final Logger logger = LoggerFactory.getLogger(IPCClient.class);

    private final int NUM_CONNECT_ATTEMPTS = 10;
    private final long CONNECT_TIMEOUT_MS = 1000;

    private UnixSocketChannel channel_;
    private LittleEndianDataOutputStream writer_;
    private LittleEndianDataInputStream reader_;
    private ObjectMapper mapper_;

    private Map<Integer, Long> mmap_table;

    public IPCClient() throws VineyardException {
        mapper_ = new ObjectMapper();
        mapper_.configure(SerializationFeature.INDENT_OUTPUT, false);
        mmap_table = new HashMap<>();
        this.connect(System.getenv("VINEYARD_IPC_SOCKET"));
    }

    public IPCClient(String ipc_socket) throws VineyardException {
        mapper_ = new ObjectMapper();
        mapper_.configure(SerializationFeature.INDENT_OUTPUT, false);
        mmap_table = new HashMap<>();
        this.connect(ipc_socket);
    }

    private synchronized void connect(String ipc_socket) throws VineyardException {
        connectIPCSocketWithRetry(ipc_socket);
        val root = mapper_.createObjectNode();
        RegisterRequest.put(root);
        this.doWrite(root);
        val reply = new RegisterReply();
        reply.get(this.doReadJson());
        this.ipc_socket = ipc_socket;
        this.rpc_endpoint = reply.getRpc_endpoint();
        this.instanceId = reply.getInstance_id();
    }

    @Override
    public ObjectMeta createMetaData(ObjectMeta meta) throws VineyardException {
        meta.setInstanceId(this.instanceId);
        meta.setTransient();
        if (!meta.hasMeta("nbytes")) {
            meta.setNBytes(0);
        }
        val root = mapper_.createObjectNode();
        CreateDataRequest.put(root, meta.metadata());
        this.doWrite(root);
        val reply = new CreateDataReply();
        reply.get(this.doReadJson());
        if (meta.isIncomplete()) {
            return getMetaData(reply.getId());
        } else {
            meta.setId(reply.getId());
            meta.setSignature(reply.getSignature());
            meta.setInstanceId(instanceId);
            return meta;
        }
    }

    @Override
    public ObjectMeta getMetaData(ObjectID id, boolean sync_remote, boolean wait)
            throws VineyardException {
        val root = mapper_.createObjectNode();
        GetDataRequest.put(root, id, sync_remote, wait);
        this.doWrite(root);
        val reply = new GetDataReply();
        reply.get(this.doReadJson());
        val contents = reply.getContents();
        if (contents.size() != 1) {
            throw new VineyardException.ObjectNotExists(
                    "Failed to read get_data_reply, size is " + contents.size());
        }

        val meta = ObjectMeta.fromMeta(contents.get(id), this.instanceId);
        val buffers = this.getBuffers(meta.getBuffers().allBufferIds());
        for (val blob : meta.getBuffers().allBufferIds()) {
            logger.debug("received blob: {}", blob);
        }
        for (val blob : meta.getBuffers().allBufferIds()) {
            if (buffers.containsKey(blob)) {
                meta.setBuffer(blob, buffers.get(blob));
            }
        }
        return meta;
    }

    @Override
    public Collection<ObjectMeta> listMetaData(String pattern) throws VineyardException {
        return listMetaData(pattern, false);
    }

    @Override
    public Collection<ObjectMeta> listMetaData(String pattern, boolean regex)
            throws VineyardException {
        return listMetaData(pattern, regex, 5);
    }

    @Override
    public Collection<ObjectMeta> listMetaData(String pattern, boolean regex, int limit)
            throws VineyardException {
        val root = mapper_.createObjectNode();
        ListDataRequest.put(root, pattern, regex, limit);
        this.doWrite(root);
        val reply = new GetDataReply();
        reply.get(this.doReadJson());
        val contents = reply.getContents();

        val metadatas = new ArrayList<ObjectMeta>();
        val bufferIds = new TreeSet<ObjectID>();
        for (val item : contents.entrySet()) {
            val meta = ObjectMeta.fromMeta(item.getValue(), this.instanceId);
            bufferIds.addAll(meta.getBuffers().allBufferIds());
            metadatas.add(meta);
        }

        val buffers = this.getBuffers(bufferIds);
        for (val meta : metadatas) {
            for (val blob : meta.getBuffers().allBufferIds()) {
                logger.debug("received blob: {}", blob);
                if (buffers.containsKey(blob)) {
                    meta.setBuffer(blob, buffers.get(blob));
                }
            }
        }
        return metadatas;
    }

    public Buffer createBuffer(long size) throws VineyardException {
        val root = mapper_.createObjectNode();
        CreateBufferRequest.put(root, size);
        this.doWrite(root);
        val reply = new CreateBufferReply();
        reply.get(this.doReadJson());

        val payload = reply.getPayload();
        long pointer = this.mmap(payload.getStoreFD(), payload.getMapSize(), true, true);
        val buffer = new Buffer();
        buffer.setObjectId(reply.getId());
        buffer.setPointer(pointer + payload.getDataOffset());
        buffer.setSize(reply.getPayload().getDataSize());
        return buffer;
    }

    private void connectIPCSocket(UnixSocketAddress address) throws VineyardException.IOError {
        try {
            channel_ = UnixSocketChannel.open(address);
        } catch (IOException e) {
            throw new VineyardException.IOError(e.getMessage());
        }
        writer_ = new LittleEndianDataOutputStream(Channels.newOutputStream(channel_));
        reader_ = new LittleEndianDataInputStream(Channels.newInputStream(channel_));
    }

    @SneakyThrows(InterruptedException.class)
    private void connectIPCSocketWithRetry(String pathname)
            throws VineyardException.ConnectionFailed {
        val address = new UnixSocketAddress(new File(pathname).getAbsolutePath());
        int num_retries = NUM_CONNECT_ATTEMPTS;
        while (num_retries > 0) {
            try {
                connectIPCSocket(address);
                break;
            } catch (VineyardException.IOError e) {
                Thread.sleep(CONNECT_TIMEOUT_MS);
            }
            num_retries -= 1;
        }
        if (reader_ == null || writer_ == null) {
            throw new VineyardException.ConnectionFailed();
        }
    }

    private Map<ObjectID, Buffer> getBuffers(Set<ObjectID> ids) throws VineyardException {
        val root = mapper_.createObjectNode();
        GetBuffersRequest.put(root, ids);
        this.doWrite(root);
        val reply = new GetBuffersReply();
        reply.get(this.doReadJson());
        Map<ObjectID, Buffer> buffers = new HashMap<>();
        for (val payload : reply.getPayloads()) {
            val buffer = new Buffer();
            if (payload.getDataSize() > 0) {
                long pointer = this.mmap(payload.getStoreFD(), payload.getMapSize(), true, true);
                buffer.setObjectId(payload.getObjectID());
                buffer.setPointer(pointer + payload.getDataOffset());
                buffer.setSize(payload.getDataSize());
            }
            buffers.put(payload.getObjectID(), buffer);
        }
        return buffers;
    }

    private long mmap(int fd, long mapSize, boolean readonly, boolean realign)
            throws VineyardException {
        if (mmap_table.containsKey(fd)) {
            return mmap_table.get(fd);
        }
        int client_fd = Fling.recvFD(this.channel_.getFD());
        long pointer = Fling.mapSharedMem(client_fd, mapSize, readonly, realign);
        if (pointer == -1) {
            throw new VineyardException.UnknownError("mmap failed for fd " + fd);
        }
        mmap_table.put(fd, pointer);
        return pointer;
    }

    private long unmap(int fd) {
        // TODO
        return -1;
    }

    @SneakyThrows(IOException.class)
    private void doWrite(String content) {
        writer_.writeLong(content.length());
        writer_.writeBytes(content);
        writer_.flush();
    }

    @SneakyThrows(JsonProcessingException.class)
    private void doWrite(JsonNode node) {
        this.doWrite(mapper_.writeValueAsString(node));
    }

    @SneakyThrows(IOException.class)
    private byte[] doRead() {
        int length = (int) reader_.readLong(); // n.b.: the server writes a size_t (long)
        val content = new byte[length];
        int done = 0, remaining = length;
        while (done < length) {
            int batch = reader_.read(content, done, remaining);
            done += batch;
            remaining -= batch;
        }
        return content;
    }

    @SneakyThrows(IOException.class)
    private JsonNode doReadJson() {
        return mapper_.readTree(doRead());
    }
}
