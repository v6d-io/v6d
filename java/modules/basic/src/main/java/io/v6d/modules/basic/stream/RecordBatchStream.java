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
package io.v6d.modules.basic.stream;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.RecordBatch;
import lombok.*;

public class RecordBatchStream extends Object {
    public static void instantiate() {
        RecordBatch.instantiate();
        ObjectFactory.getFactory()
                .register("vineyard::RecordBatchStream", new RecordBatchStreamResolver());
    }

    public static class Reader {
        private final Client client;
        private final ObjectID stream;

        public Reader(final Client client, final ObjectID stream) throws VineyardException {
            this.client = client;
            this.stream = stream;
            this.client.openStream(this.stream, 'r');
        }

        public ObjectID nextChunkID() throws VineyardException {
            return this.client.pullStreamChunkID(this.stream);
        }

        public ObjectMeta nextChunkMeta() throws VineyardException {
            return this.client.pullStreamChunkMeta(this.stream);
        }

        public RecordBatch nextChunk() throws VineyardException {
            return (RecordBatch) ObjectFactory.getFactory().resolve(this.nextChunkMeta());
        }
    }

    public static class Writer {
        private final Client client;
        private final ObjectID stream;

        public Writer(final Client client, final ObjectID stream) throws VineyardException {
            this.client = client;
            this.stream = stream;
            this.client.openStream(this.stream, 'w');
        }

        public void append(final ObjectID chunk) throws VineyardException {
            this.client.pushStreamChunk(this.stream, chunk);
        }

        public void fail() throws VineyardException {
            this.client.stopStream(this.stream, true);
        }

        public void finish() throws VineyardException {
            this.client.stopStream(this.stream, false);
        }
    }

    private final ObjectID id;
    private Reader reader;
    private Writer writer;

    public RecordBatchStream(final ObjectMeta meta) {
        super(meta);
        this.id = meta.getId();
    }

    public static RecordBatchStream create(IPCClient client) throws VineyardException {
        var meta = ObjectMeta.empty();
        meta.setTypename("vineyard::RecordBatchStream");
        meta = client.createMetaData(meta);
        client.createStream(meta.getId());
        return new RecordBatchStream(meta);
    }

    public ObjectID getId() {
        return id;
    }

    public Reader reader(final Client client) throws VineyardException {
        if (this.reader == null) {
            this.reader = new Reader(client, this.id);
        }
        return this.reader;
    }

    public Writer writer(final Client client) throws VineyardException {
        if (this.writer == null) {
            this.writer = new Writer(client, this.id);
        }
        return this.writer;
    }
}

class RecordBatchStreamResolver extends ObjectFactory.Resolver {

    @Override
    public Object resolve(ObjectMeta metadata) {
        return new RecordBatchStream(metadata);
    }
}
