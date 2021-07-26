/** Copyright 2020-2021 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package io.v6d.core.client;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.io.LittleEndianDataInputStream;
import com.google.common.io.LittleEndianDataOutputStream;
import io.v6d.core.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.Protocol.*;
import io.v6d.core.common.util.VineyardException;
import java.io.*;
import java.nio.channels.Channels;
import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;
import lombok.*;

/** Vineyard IPC client. */
public class IPCClient extends Client {
    private UnixSocketChannel channel_;
    private LittleEndianDataOutputStream writer_;
    private LittleEndianDataInputStream reader_;
    private ObjectMapper mapper_;

    private final int NUM_CONNECT_ATTEMPTS = 10;
    private final long CONNECT_TIMEOUT_MS = 1000;

    public IPCClient() throws VineyardException {
        mapper_ = new ObjectMapper();
        this.Connect(System.getenv("VINEYARD_IPC_SOCKET"));
    }

    public IPCClient(String ipc_socket) throws VineyardException {
        mapper_ = new ObjectMapper();
        this.Connect(ipc_socket);
    }

    private synchronized void Connect(String ipc_socket) throws VineyardException {
        connectIPCSocketWithRetry(ipc_socket);
        var root = mapper_.createObjectNode();
        var req = new RegisterRequest();
        req.Put(root);
        this.doWrite(root);
        var reply = new RegisterReply();
        reply.Get(this.doReadJson());
        this.ipc_socket = ipc_socket;
        this.rpc_endpoint = reply.getRpc_endpoint();
    }

    @Override
    public ObjectID CreateMetaData(ObjectMeta metadata) {
        return null;
    }

    @Override
    public ObjectMeta GetMetaData(ObjectID id, boolean sync_remote) {
        return null;
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
        UnixSocketAddress address = new UnixSocketAddress(new File(pathname).getAbsolutePath());
        int num_retries = NUM_CONNECT_ATTEMPTS;
        while (num_retries > 0) {
            try {
                connectIPCSocket(address);
                break;
            } catch (VineyardException.IOError e) {
                e.printStackTrace();
                Thread.sleep(CONNECT_TIMEOUT_MS);
            }
            num_retries += 1;
        }
        if (reader_ == null || writer_ == null) {
            throw new VineyardException.ConnectionFailed();
        }
    }

    @SneakyThrows(IOException.class)
    private void doWrite(String content) {
        writer_.writeLong((long) content.length());
        writer_.writeChars(content);
        writer_.flush();
    }

    @SneakyThrows(JsonProcessingException.class)
    private void doWrite(JsonNode node) {
        this.doWrite(mapper_.writeValueAsString(node));
    }

    @SneakyThrows(IOException.class)
    private byte[] doRead() {
        int length = (int) reader_.readLong(); // n.b.: the server writes a size_t (long)
        byte[] content = new byte[(int) length];
        reader_.read(content, 0, length);
        return content;
    }

    @SneakyThrows(IOException.class)
    private JsonNode doReadJson() {
        byte[] content = doRead();
        return mapper_.readTree(content);
    }
}
