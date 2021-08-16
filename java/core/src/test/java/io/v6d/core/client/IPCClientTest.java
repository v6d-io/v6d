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

import io.v6d.core.client.IPCClient;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import lombok.*;
import org.junit.Test;

/** Unit test for IPC client. */
public class IPCClientTest {
    @Test
    public void connect() throws VineyardException {
       val client = new IPCClient();
       val meta = client.getMetaData(ObjectID.fromString("o000046b523f6a53a"));

       System.out.println(meta);
    }
}
