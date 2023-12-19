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

    private UnixSocketChannel channel;
    private LittleEndianDataOutputStream writer;
    private LittleEndianDataInputStream reader;
    private ObjectMapper mapper;

    private Map<Integer, Long> mmapTable;

    public IPCClient() throws VineyardException {
        mapper = new ObjectMapper();
        mapper.configure(SerializationFeature.INDENT_OUTPUT, false);
        mmapTable = new HashMap<>();
        val ipc_socket = System.getenv(Client.DEFAULT_IPC_SOCKET_KEY);
        if (ipc_socket == null) {
            throw new VineyardException.ConnectionFailed(
                    "Failed to resolve default vineyard IPC socket, "
                            + "please make sure the environment variable "
                            + Client.DEFAULT_IPC_SOCKET_KEY
                            + " is set.");
        }
        this.connect(ipc_socket);
    }

    public IPCClient(String ipc_socket) throws VineyardException {
        mapper = new ObjectMapper();
        mapper.configure(SerializationFeature.INDENT_OUTPUT, false);
        mmapTable = new HashMap<>();
        this.connect(ipc_socket);
    }

    private synchronized void connect(String ipc_socket) throws VineyardException {
        connectIPCSocketWithRetry(ipc_socket);
        val root = mapper.createObjectNode();
        RegisterRequest.put(root);
        this.doWrite(root);
        val reply = new RegisterReply();
        reply.get(this.doReadJson());
        this.ipc_socket = ipc_socket;
        this.rpc_endpoint = reply.getRpc_endpoint();
        this.instanceId = reply.getInstance_id();
    }

    @Override
    public synchronized ObjectMeta createMetaData(ObjectMeta meta) throws VineyardException {
        meta.setInstanceId(this.instanceId);
        meta.setTransient();
        if (!meta.hasMeta("nbytes")) {
            meta.setNBytes(0);
        }
        val root = mapper.createObjectNode();
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
    public boolean connected() {
        return channel != null && channel.isConnected();
    }

    @Override
    public synchronized ObjectMeta getMetaData(
            ObjectID id, boolean migrate, boolean sync_remote, boolean wait)
            throws VineyardException {
        ObjectMeta meta = getMetaDataInternal(id, sync_remote, wait);
        if (meta.getInstanceId().compareTo(this.instanceId) != 0 && migrate && (!meta.isGlobal())) {
            return getMetaDataInternal(this.migrateObject(id), sync_remote, wait);
        }
        return meta;
    }

    private synchronized ObjectMeta getMetaDataInternal(
            ObjectID id, boolean sync_remote, boolean wait) throws VineyardException {
        val root = mapper.createObjectNode();
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
            if (buffers.containsKey(blob)) {
                meta.setBuffer(blob, buffers.get(blob));
            }
        }
        return meta;
    }

    @Override
    public synchronized ObjectID migrateObject(ObjectID id) throws VineyardException {
        val root = mapper.createObjectNode();
        MigrateObjectRequest.put(root, id);
        this.doWrite(root);
        val reply = new MigrateObjectReply();
        reply.get(this.doReadJson());
        return reply.getObjectID();
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
    public synchronized Collection<ObjectMeta> listMetaData(
            String pattern, boolean regex, int limit) throws VineyardException {
        val root = mapper.createObjectNode();
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
                if (buffers.containsKey(blob)) {
                    meta.setBuffer(blob, buffers.get(blob));
                }
            }
        }
        return metadatas;
    }

    @Override
    public synchronized Map<String, ObjectID> listNames(String pattern, boolean regex, int limit)
            throws VineyardException {
        val root = mapper.createObjectNode();
        ListNameRequest.put(root, pattern, regex, limit);
        this.doWrite(root);
        val reply = new ListNameReply();
        reply.get(this.doReadJson());
        val contents = reply.getContents();

        Map<String, ObjectID> result = new HashMap<>();
        for (val item : contents.entrySet()) {
            result.put(item.getKey(), item.getValue());
        }

        return result;
    }

    @Override
    public synchronized void persist(ObjectID id) throws VineyardException {
        val root = mapper.createObjectNode();
        PersistRequest.put(root, id);
        this.doWrite(root);
        val reply = new PersistReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void delete(Collection<ObjectID> ids, boolean force, boolean deep)
            throws VineyardException {
        val root = mapper.createObjectNode();
        DeleteDataRequest.put(root, ids, force, deep, false);
        this.doWrite(root);
        val reply = new DeleteDataReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void putName(ObjectID id, String name) throws VineyardException {
        val root = mapper.createObjectNode();
        PutNameRequest.put(root, id, name);
        this.doWrite(root);
        val reply = new PutNameReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized ObjectID getName(String name, boolean wait) throws VineyardException {
        val root = mapper.createObjectNode();
        GetNameRequest.put(root, name, wait);
        this.doWrite(root);
        val reply = new GetNameReply();
        reply.get(this.doReadJson());
        return reply.getId();
    }

    @Override
    public synchronized void dropName(String name) throws VineyardException {
        val root = mapper.createObjectNode();
        DropNameRequest.put(root, name);
        this.doWrite(root);
        val reply = new DropNameReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void createStream(final ObjectID id) throws VineyardException {
        val root = mapper.createObjectNode();
        CreateStreamRequest.put(root, id);
        this.doWrite(root);
        val reply = new CreateStreamReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void openStream(final ObjectID id, final char mode)
            throws VineyardException {
        val root = mapper.createObjectNode();
        VineyardException.asserts(
                mode == 'r' || mode == 'w', "invalid mode to open stream: '" + mode + "'");
        OpenStreamRequest.put(root, id, mode == 'r' ? 1 : 2);
        this.doWrite(root);
        val reply = new OpenStreamReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void pushStreamChunk(final ObjectID id, final ObjectID chunk)
            throws VineyardException {
        val root = mapper.createObjectNode();
        PushNextStreamChunkRequest.put(root, id, chunk);
        this.doWrite(root);
        val reply = new PushNextStreamChunkReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized ObjectID pullStreamChunkID(final ObjectID id) throws VineyardException {
        val root = mapper.createObjectNode();
        PullNextStreamChunkRequest.put(root, id);
        this.doWrite(root);
        val reply = new PullNextStreamChunkReply();
        reply.get(this.doReadJson());
        return reply.getChunk();
    }

    @Override
    public synchronized ObjectMeta pullStreamChunkMeta(final ObjectID id) throws VineyardException {
        val chunk = this.pullStreamChunkID(id);
        return this.getMetaData(chunk);
    }

    @Override
    public synchronized void stopStream(final ObjectID id, boolean failed)
            throws VineyardException {
        val root = mapper.createObjectNode();
        StopStreamRequest.put(root, id, failed);
        this.doWrite(root);
        val reply = new StopStreamReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized InstanceStatus getInstanceStatus() throws VineyardException {
        val root = mapper.createObjectNode();
        InstanceStatusRequest.put(root);
        this.doWrite(root);
        val reply = new InstanceStatusReply();
        reply.get(this.doReadJson());
        return InstanceStatus.fromJson(reply.getStatus());
    }

    @Override
    public synchronized ClusterStatus getClusterStatus() throws VineyardException {
        val root = mapper.createObjectNode();
        ClusterStatusRequest.put(root);
        this.doWrite(root);
        val reply = new ClusterStatusReply();
        reply.get(this.doReadJson());
        return ClusterStatus.fromJson(reply.getStatus());
    }

    public synchronized Buffer createBuffer(long size) throws VineyardException {
        if (size == 0) {
            return Buffer.empty();
        }
        val root = mapper.createObjectNode();
        CreateBufferRequest.put(root, size);
        this.doWrite(root);
        val reply = new CreateBufferReply();
        reply.get(this.doReadJson());

        val payload = reply.getPayload();
        long pointer = this.mmap(payload.getStoreFD(), payload.getMapSize(), false, true);
        val buffer = new Buffer();
        buffer.setObjectId(reply.getId());
        buffer.setPointer(pointer + payload.getDataOffset());
        buffer.setSize(reply.getPayload().getDataSize());
        return buffer;
    }

    @Override
    public synchronized void sealBuffer(ObjectID objectID) throws VineyardException {
        VineyardException.asserts(objectID.isBlob(), "Not a blob object id: " + objectID);

        val root = mapper.createObjectNode();
        SealBufferRequest.put(root, objectID);
        this.doWrite(root);
        val reply = new SealBufferReply();
        reply.get(this.doReadJson());
    }

    @Override
    public synchronized void shrinkBuffer(ObjectID objectID, long size) throws VineyardException {
        VineyardException.asserts(objectID.isBlob(), "Not a blob object id: " + objectID);

        val root = mapper.createObjectNode();
        ShrinkBufferRequest.put(root, objectID, size);
        this.doWrite(root);
        val reply = new ShrinkBufferReply();
        reply.get(this.doReadJson());
    }

    private void connectIPCSocket(UnixSocketAddress address) throws VineyardException.IOError {
        try {
            channel = UnixSocketChannel.open(address);
        } catch (IOException e) {
            throw new VineyardException.IOError(e.getMessage());
        }
        writer = new LittleEndianDataOutputStream(Channels.newOutputStream(channel));
        reader = new LittleEndianDataInputStream(Channels.newInputStream(channel));
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

        if (reader == null || writer == null) {
            throw new VineyardException.ConnectionFailed();
        }
    }

    private Map<ObjectID, Buffer> getBuffers(Set<ObjectID> ids) throws VineyardException {
        val root = mapper.createObjectNode();
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
        if (mmapTable.containsKey(fd)) {
            return mmapTable.get(fd);
        }
        int client_fd = Fling.recvFD(this.channel.getFD());
        long pointer = Fling.mapSharedMem(client_fd, mapSize, readonly, realign);
        if (pointer == -1) {
            throw new VineyardException.UnknownError("mmap failed for fd " + fd);
        }
        mmapTable.put(fd, pointer);
        return pointer;
    }

    private long unmap(int fd) {
        // TODO
        return -1;
    }

    @SneakyThrows(IOException.class)
    private void doWrite(String content) {
        writer.writeLong(content.length());
        writer.writeBytes(content);
        writer.flush();
    }

    @SneakyThrows(JsonProcessingException.class)
    private void doWrite(JsonNode node) {
        this.doWrite(mapper.writeValueAsString(node));
    }

    @SneakyThrows(IOException.class)
    private byte[] doRead() {
        int length = (int) reader.readLong(); // n.b.: the server writes a size_t (long)
        val content = new byte[length];
        int done = 0, remaining = length;
        while (done < length) {
            int batch = reader.read(content, done, remaining);
            done += batch;
            remaining -= batch;
        }
        return content;
    }

    @SneakyThrows(IOException.class)
    private JsonNode doReadJson() {
        return mapper.readTree(doRead());
    }
}
