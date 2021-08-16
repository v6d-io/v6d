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

/** Vineyard IPC client.s */
public abstract class Client {
    protected String ipc_socket;
    protected String rpc_endpoint;

    public abstract ObjectID CreateMetaData(ObjectMeta metadata);

    public ObjectMeta GetMetaData(ObjectID id) {
        return this.GetMetaData(id, false);
    }

    public abstract ObjectMeta GetMetaData(ObjectID id, boolean sync_remote);

    public boolean Connected() {
        return false;
    }

    public void Disconnect() {}

    public String IPCSocket() {
        return ipc_socket;
    }

    public String RPCEndpoint() {
        return rpc_endpoint;
    }
}
