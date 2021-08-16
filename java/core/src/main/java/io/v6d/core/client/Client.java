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

import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;

/** Vineyard IPC client. */
public abstract class Client {
    protected String ipc_socket;
    protected String rpc_endpoint;

    public abstract ObjectID createMetaData(ObjectMeta metadata) throws VineyardException;

    public ObjectMeta getMetaData(ObjectID id) throws VineyardException {
        return this.getMetaData(id, false);
    }

    public ObjectMeta getMetaData(ObjectID id, boolean sync_remote) throws VineyardException {
        return this.getMetaData(id, sync_remote, false);
    }

    public abstract ObjectMeta getMetaData(ObjectID id, boolean sync_remote, boolean wait) throws VineyardException;

    public boolean connected() {
        return false;
    }

    public void disconnect() {}

    public String getIPCSocket() {
        return ipc_socket;
    }

    public String getRPCEndpoint() {
        return rpc_endpoint;
    }
}
