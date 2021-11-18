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

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.VineyardException;
import lombok.val;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.junit.Before;
import org.junit.Test;

/** Unit test for arrow data structures. */
public class ArrowTest {
    private IPCClient client;

    @Before
    public void prepareResolvers() throws VineyardException {
        client = new IPCClient();

        Arrow.instantiate();
    }

    @Test
    public void testDoubleArray() throws VineyardException {
        val builder = new DoubleArrayBuilder(client, 5);
        builder.set(0, 1);
        builder.set(1, 2);
        builder.set(2, 3);
        builder.set(3, 4);
        builder.set(4, 5);

        val meta = builder.seal(client);
        assertEquals(5, meta.getIntValue("length_"));

        val array =
                (DoubleArray) ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));
        assertEquals(5, array.length());

        val expected = builder.columnar();
        val actual = array.columnar();
        assertEquals(expected.valueCount(), actual.valueCount());
        for (int index = 0; index < array.length(); ++index) {
            assertEquals(expected.getDouble(index), actual.getDouble(index), 0.001);
        }
    }

    @Test
    public void testSchema() throws VineyardException {
        val builder = new SchemaBuilder();
        builder.addField(Arrow.makePlainField("testa", Arrow.FieldType.Int));
        builder.addField(Arrow.makePlainField("testb", Arrow.FieldType.Double));
        builder.addField(Arrow.makePlainField("teststring", Arrow.FieldType.Double));
        builder.addMetadata("kind", "testmeta");
        val meta = builder.seal(client);

        val schema = (Schema) ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));
        assertNotNull(schema);
        val arrowSchema = schema.getSchema();
        assertEquals(arrowSchema.getFields().size(), 3);
        assertEquals(arrowSchema.getCustomMetadata().size(), 1);
        assertEquals(arrowSchema.getFields().get(0).getName(), "testa");
        assertEquals(arrowSchema.getFields().get(1).getName(), "testb");
        assertEquals(arrowSchema.getFields().get(2).getName(), "teststring");
        assertEquals(arrowSchema.getCustomMetadata().get("kind"), "testmeta");
    }

    @Test
    public void testRecordBatch() throws VineyardException {
        val builder = new RecordBatchBuilder(client, 5);
        builder.addField(Arrow.makePlainField("testa", Arrow.FieldType.Int));
        builder.addField(Arrow.makePlainField("testb", Arrow.FieldType.Double));
        builder.addCustomMetadata("kind", "testbatch");
        builder.finishSchema(client);

        val column0 = builder.getColumnBuilder(0);
        column0.setInt(0, 1);
        column0.setInt(1, 2);
        column0.setInt(2, 3);
        column0.setInt(3, 4);
        column0.setInt(4, 5);

        val column1 = builder.getColumnBuilder(1);
        column1.setDouble(0, 100.1);
        column1.setDouble(1, 200.2);
        column1.setDouble(2, 300.3);
        column1.setDouble(3, 400.4);
        column1.setDouble(4, 500.5);

        val meta = builder.seal(client);

        val batch =
                (RecordBatch) ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));

        val array0 = (IntVector) batch.getBatch().getVector(0);
        assertEquals(5, array0.getValueCount());
        assertEquals(1, array0.get(0));
        assertEquals(2, array0.get(1));
        assertEquals(3, array0.get(2));
        assertEquals(4, array0.get(3));
        assertEquals(5, array0.get(4));

        val array1 = (Float8Vector) batch.getBatch().getVector(1);
        assertEquals(5, array1.getValueCount());
        assertEquals(100.1, array1.get(0), 0.001);
        assertEquals(200.2, array1.get(1), 0.001);
        assertEquals(300.3, array1.get(2), 0.001);
        assertEquals(400.4, array1.get(3), 0.001);
        assertEquals(500.5, array1.get(4), 0.001);

        assertEquals("testbatch", batch.getBatch().getSchema().getCustomMetadata().get("kind"));
    }
}
