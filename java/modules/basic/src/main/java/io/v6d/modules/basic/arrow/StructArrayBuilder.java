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

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.util.ArrowVectorUtils;
import java.util.ArrayList;
import java.util.List;
import lombok.val;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

public class StructArrayBuilder implements ArrayBuilder {
    private BufferBuilder[] bufferBuilders;
    private StructVector array;
    private List<Integer> valueCountList;
    private SchemaBuilder structVectorSchemaBuilder;

    public StructArrayBuilder(IPCClient client, Field field) throws VineyardException {
        this.array = StructVector.empty("", Arrow.default_allocator);
        ArrowVectorUtils.buildArrowVector(this.array, field);

        List<Field> fields = new ArrayList<>();
        fields.add(field);
        Schema schema = new Schema(fields);
        structVectorSchemaBuilder = SchemaBuilder.fromSchema(schema);
    }

    @Override
    public void build(Client client) throws VineyardException {
        valueCountList = ArrowVectorUtils.getValueCountOfArrowVector(array);
        ArrowBuf[] buffers = ArrowVectorUtils.getArrowBuffers(array);

        this.bufferBuilders = new BufferBuilder[buffers.length];
        for (int i = 0; i < buffers.length; i++) {
            this.bufferBuilders[i] =
                    new BufferBuilder((IPCClient) client, buffers[i], buffers[i].capacity());
        }
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();

        meta.setTypename("vineyard::StructArray");
        meta.setValue("bufs_num_", this.bufferBuilders.length);
        for (int i = 0; i < this.bufferBuilders.length; i++) {
            meta.addMember(
                    "buffer_" + String.valueOf(i) + "_", this.bufferBuilders[i].seal(client));
        }

        meta.setValue("value_count_num_", valueCountList.size());
        for (int i = 0; i < valueCountList.size(); i++) {
            meta.setValue("value_count_" + String.valueOf(i) + "_", valueCountList.get(i));
        }

        meta.addMember("schema_", structVectorSchemaBuilder.seal(client));
        return client.createMetaData(meta);
    }

    @Override
    public FieldVector getArray() {
        return this.array;
    }

    @Override
    public void shrink(Client client, long size) throws VineyardException {
        this.array.setValueCount((int) size);
    }

    void set(int index, ValueVector value) {
        this.array.copyFromSafe(0, index, value);
    }
}
