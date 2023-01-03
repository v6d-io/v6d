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
package io.v6d.modules.basic.arrow;

import static org.junit.Assert.assertEquals;

import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.VineyardException;
import lombok.val;
import org.junit.Before;
import org.junit.Test;

/** Unit test for buffer and blob APIs. */
public class BufferTest {
    private IPCClient client;

    @Before
    public void prepareResolvers() throws VineyardException {
        client = new IPCClient();

        Arrow.instantiate();
    }

    @Test
    public void testBlob() throws VineyardException {
        val builder = new BufferBuilder(client, 40);
        val meta = builder.seal(client);

        val buffer = (Buffer) ObjectFactory.getFactory().resolve(meta);
        assertEquals(40, buffer.length());
    }

    @Test
    public void testGetBlob() throws VineyardException {
        val builder = new BufferBuilder(client, 40);
        val meta = builder.seal(client);

        val buffer = (Buffer) ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));
        assertEquals(40, buffer.length());
    }
}
