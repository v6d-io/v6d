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
import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.core.common.util.VineyardException.NotImplemented;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import lombok.val;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

public class ListArrayBuilder implements ArrayBuilder {
    private BufferBuilder offset_buffer_builder;
    private BufferBuilder []data_buffer_builder;
    private ListVector array;
    private Field field;
    private int bitWidth;
    private int length;
    // private List<ArrayBuilder> arrayBuilders;

    public ListArrayBuilder(IPCClient client, long length, Field childField) throws VineyardException {
        Context.println("childField:" + childField.getType().getTypeID().name());
        this.field = childField;
        this.array = new ListVector("", Arrow.default_allocator, childField.getFieldType(), null);
        this.array.addOrGetVector(childField.getFieldType());
        switch(childField.getType().getTypeID()) {
            case Int:
                this.bitWidth = ((ArrowType.Int)childField.getType()).getBitWidth();
                this.length = (int)length;
                break;
                // if (((ArrowType.Int)childField.getType()).getBitWidth() == 8) {
                //     //tinyint
                //     for (int i = 0; i < length; i++) {
                //         this.arrayBuilders.add(new Int8ArrayBuilder(client, 0));
                //     }
                // } else if (((ArrowType.Int)childField.getType()).getBitWidth() == 16) {
                //     //smallint
                //     for (int i = 0; i < length; i++) {
                //         this.arrayBuilders.add(new Int16ArrayBuilder(client, 0));
                //     }
                // } else if (((ArrowType.Int)childField.getType()).getBitWidth() == 32) {
                //     //int
                //     Context.println("int32!!!!!!");
                //     for (int i = 0; i < length; i++) {
                //         this.arrayBuilders.add(new Int32ArrayBuilder(client, 0));
                //     }
                // } else if (((ArrowType.Int)childField.getType()).getBitWidth() == 64) {
                //     //bigint
                //     for (int i = 0; i < length; i++) {
                //         this.arrayBuilders.add(new Int64ArrayBuilder(client, 0));
                //     }
                // } else {
                //     throw new NotImplemented("Unsupported type: " + childField.getType().getTypeID().name());
                // }
            default:
                throw new NotImplemented("Unsupported type: " + childField.getType().getTypeID().name());
        }
    }

    @Override
    public void build(Client client) throws VineyardException {
        array.setValueCount(length);
        long dataLength = this.array.getDataVector().getBufferSize();
        Context.println("dataLength:" + dataLength);
        long offsetLength = (this.array.getValueCount() + 1) * this.array.OFFSET_WIDTH;
        
        // this.array.getDataBuffer()
        // this.array.getValidityBuffer()
        // this.array.getOffsetBuffer()
        // long offset_length = (this.array.getValueCount() + 1) * ListVector.OFFSET_WIDTH;
        // this.array.getBuffers(false)
        // this.array.getOffsetBuffer();
        // this.buffer = new BufferBuilder((IPCClient)client, length);
        val offset_buffer = this.array.getOffsetBuffer();
        // val data_buffer = this.array.getDataBuffer();
        this.offset_buffer_builder = new BufferBuilder((IPCClient)client, offset_buffer, offsetLength);
        ArrowBuf[] data_buffer = this.array.getBuffers(false);
        // offset buffer, valid buffer, data buffer
        this.data_buffer_builder = new BufferBuilder[data_buffer.length - 2];
        for (int i = 0; i < data_buffer_builder.length; i++) {
            Context.println("data_buffer[" + i + "]:" + data_buffer[i + 2].toString());
            this.data_buffer_builder[i] = new BufferBuilder((IPCClient)client, data_buffer[i + 2], data_buffer[i + 2].capacity());
        }
    }

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        val meta = ObjectMeta.empty();
        // meta.setTypename("vineyard::ListArray<" + field.getType().toString() + ">");
        // Context.println("vineyard::ListArray<" + field.getType().toString() + ">");
        meta.setTypename("vineyard::ListArray");
        meta.setNBytes(array.getBufferSizeFor(array.getValueCount()));
        meta.setValue("length_", array.getValueCount());
        meta.setValue("listLength_", this.data_buffer_builder.length);
        meta.setValue("null_count_", 0);
        meta.setValue("offset_", 0);
        for (int i = 0; i < this.data_buffer_builder.length; i++) {
            meta.addMember("buffer_" + String.valueOf(i) + "_", this.data_buffer_builder[i].seal(client));
        }
        meta.addMember("buffer_offsets_", offset_buffer_builder.seal(client));
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
        meta.setValue("bitWidth_", bitWidth);
        Context.println("bitWidth:" + bitWidth);
        // meta.addMember("buffer_", buffer.seal(client));
        // for (int i = 0; i < this.arrayBuilders.size(); i++) {
        //     meta.addMember("children_" + String.valueOf(i) + "_", this.arrayBuilders.get(i).seal(client));
        // }
        meta.addMember("null_bitmap_", BufferBuilder.empty(client));
        meta.setValue("valueType_", field.getType().getTypeID().ordinal());
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
