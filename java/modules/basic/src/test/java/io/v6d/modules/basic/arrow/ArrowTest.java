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
package io.v6d.modules.basic.arrow;

import static io.v6d.modules.basic.arrow.Arrow.logger;
import static org.junit.Assert.*;

import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.Env;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import lombok.val;
import org.junit.Before;
import org.junit.Test;

/** Unit test for simple App. */
public class ArrowTest {
    private IPCClient client;
    private final String testBlobID = Env.getEnvOrNull("TEST_BLOB_ID");
    private final String testDoubleArrayID = Env.getEnvOrNull("TEST_DOUBLE_ARRAY_ID");
    private final String testRecordBatchID = Env.getEnvOrNull("TEST_RECORD_BATCH_ID");

    @Before
    public void prepareResolvers() throws VineyardException {
        Arrow.instantiate();
        client = new IPCClient();
    }

    @Test
    public void testBlob() throws VineyardException {
        val meta = client.getMetaData(ObjectID.fromString(testBlobID));
        logger.debug("metadata = {}", meta);

        val buffer = (Buffer) ObjectFactory.getFactory().resolve(meta);
        assertEquals(40, buffer.length());
    }

    @Test
    public void testDoubleArray() throws VineyardException {
        val meta = client.getMetaData(ObjectID.fromString(testDoubleArrayID));
        logger.debug("metadata = {}", meta);

        val array = (DoubleArray) ObjectFactory.getFactory().resolve(meta);
        logger.debug("array is: {}", array.getArray());
        assertEquals(5, array.getArray().getValueCount());
    }

    @Test
    public void testRecordBatch() throws VineyardException {
        val meta = client.getMetaData(ObjectID.fromString(testRecordBatchID));
        logger.debug("metadata = {}", meta);

        val batch = (RecordBatch) ObjectFactory.getFactory().resolve(meta);
        logger.debug("batch is: {}", batch);
    }
}
