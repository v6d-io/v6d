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

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;

import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.VineyardException;
import lombok.val;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.LargeVarCharVector;
import org.apache.arrow.vector.VarCharVector;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

/** Unit test for arrow data structures. */
public class ArrowTest {
    private IPCClient client;

    @Before
    @Ignore
    public void prepareResolvers() throws VineyardException {
        client = new IPCClient();

        Arrow.instantiate();
    }

    @Test
    @Ignore
    public void testBooleanArray() throws VineyardException {
        val builder = new BooleanArrayBuilder(client, 5);
        builder.set(0, true);
        builder.set(1, false);
        builder.set(2, true);
        builder.set(3, false);
        builder.set(4, true);

        val meta = builder.seal(client);
        assertEquals(5, meta.getIntValue("length_"));
        val array = (BooleanArray) ObjectFactory.getFactory().resolve(meta);
        assertEquals(5, array.length());

        val expected = builder.columnar();
        val actual = array.columnar();
        assertEquals(expected.valueCount(), actual.valueCount());
        for (int index = 0; index < array.length(); ++index) {
            assertEquals(expected.getBoolean(index), actual.getBoolean(index));
        }
    }

    @Test
    @Ignore
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
    @Ignore
    public void testStringArray() throws VineyardException {
        val base = new VarCharVector("", Arrow.default_allocator);
        base.setSafe(0, "hello".getBytes(), 0, 5);
        base.setSafe(1, " ".getBytes(), 0, 1);
        base.setSafe(2, "world".getBytes(), 0, 5);
        base.setValueCount(3); // nb. important

        val builder = new StringArrayBuilder(client, base);
        val meta = builder.seal(client);
        assertEquals(3, meta.getIntValue("length_"));

        val array =
                (StringArray) ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));
        assertEquals(3, array.length());

        val expected = builder.columnar();
        val actual = array.columnar();
        assertEquals(expected.valueCount(), actual.valueCount());
        for (int index = 0; index < array.length(); ++index) {
            assertEquals(expected.getUTF8String(index), actual.getUTF8String(index));
        }
    }

    @Test
    @Ignore
    public void testLargeStringArray() throws VineyardException {
        val base = new LargeVarCharVector("", Arrow.default_allocator);
        base.setSafe(0, "hello".getBytes(), 0, 5);
        base.setSafe(1, " ".getBytes(), 0, 1);
        base.setSafe(2, "world".getBytes(), 0, 5);
        base.setValueCount(3); // nb. important

        val builder = new LargeStringArrayBuilder(client, base);
        val meta = builder.seal(client);
        assertEquals(3, meta.getIntValue("length_"));

        val array =
                (LargeStringArray)
                        ObjectFactory.getFactory().resolve(client.getMetaData(meta.getId()));
        assertEquals(3, array.length());

        val expected = builder.columnar();
        val actual = array.columnar();
        assertEquals(expected.valueCount(), actual.valueCount());
        for (int index = 0; index < array.length(); ++index) {
            assertEquals(expected.getUTF8String(index), actual.getUTF8String(index));
        }
    }

    @Test
    @Ignore
    public void testSchema() throws VineyardException {
        val builder = new SchemaBuilder();
        builder.addField(Arrow.makeField("testa", Arrow.FieldType.Int));
        builder.addField(Arrow.makeField("testb", Arrow.FieldType.Double));
        builder.addField(Arrow.makeField("teststring", Arrow.FieldType.Double));
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
    @Ignore
    public void testRecordBatch() throws VineyardException {
        val builder = new RecordBatchBuilder(client, 5);
        builder.addField(Arrow.makeField("testa", Arrow.FieldType.Int));
        builder.addField(Arrow.makeField("testb", Arrow.FieldType.Double));
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
