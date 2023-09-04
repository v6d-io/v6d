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

import static org.junit.Assert.*;

import com.google.common.collect.ImmutableList;
import io.v6d.core.client.ds.*;
import io.v6d.core.common.util.VineyardException;
import lombok.val;
import org.junit.Before;
import org.junit.Test;

/** Unit test for IPC client. */
public class IPCClientTest {
    private IPCClient client;

    @Before
    public void prepareResolvers() throws VineyardException {
        client = new IPCClient();

        val factory = ObjectFactory.getFactory();
        factory.register("vineyard::Scalar<std::string>", new StringObjectResolver());
    }

    @Test
    public void testConnect() throws VineyardException {
        val client = new IPCClient();
        System.out.println("client = " + client);
    }

    @Test
    public void testCreateAndGetMetadata() throws VineyardException {
        val builder = new StringObjectBuilder("abcde");
        val result = builder.seal(client);
        val meta = client.getMetaData(result.getId());
        val value = (StringObject) ObjectFactory.getFactory().resolve(meta);

        assertEquals(builder.getStr(), value.getStr());
    }

    @Test
    public void testListData() throws VineyardException {
        val builder = new StringObjectBuilder("abcde");
        val result = builder.seal(client);
        val meta = client.getMetaData(result.getId());
        val value = (StringObject) ObjectFactory.getFactory().resolve(meta);

        val metadatas = client.listMetaData("vineyard::Scalar<std::string>", false, 1024);
        assertFalse(metadatas.isEmpty());
        for (val metadata : metadatas) {
            assertEquals(metadata.getTypename(), "vineyard::Scalar<std::string>");
        }
    }

    @Test
    public void testPersist() throws VineyardException {
        val builder = new StringObjectBuilder("abcde");
        val result = builder.seal(client);
        assertFalse(result.isPersist());
        client.persist(result);

        val target = client.getMetaData(result.getId());
        assertTrue(target.isPersist());
    }

    @Test
    public void testDelete() throws VineyardException {
        val builder = new StringObjectBuilder("abcde");
        val result = builder.seal(client);

        // `get` should be ok
        client.getMetaData(result.getId());

        // `delete` the object
        client.delete(ImmutableList.of(result.getId()));

        // `get` will throw error
        assertThrows(
                VineyardException.ObjectNotExists.class,
                () -> {
                    client.getMetaData(result.getId());
                });
    }

    @Test
    public void testNames() throws VineyardException {
        val builder = new StringObjectBuilder("abcde");
        val result = builder.seal(client);

        assertThrows(
                VineyardException.Invalid.class,
                () -> {
                    client.putName(result.getId(), "test_name");
                });

        client.persist(result);
        client.putName(result.getId(), "test_name");

        assertEquals(result.getId(), client.getName("test_name"));
        val target = client.getMetaData(result.getId());
        assertTrue(target.hasName());
        assertEquals("test_name", target.getName());
    }
}
