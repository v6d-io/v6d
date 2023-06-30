package io.v6d.modules.basic.tensor;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBase;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.Buffer;
import io.v6d.modules.basic.arrow.BufferBuilder;
import io.v6d.modules.basic.arrow.Int32ArrayBuilder;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.TableBuilder;

import java.util.ArrayList;
import java.util.Collection;
import java.lang.Long;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.types.Types;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TensorBuilder implements ObjectBuilder{
    private FieldVector values;
    private BufferBuilder bufferBuilder;
    private Collection<Integer> shape;
    private Types.MinorType dtype;
    private int partition_index_ = 0;
    private long size = 0;

    public TensorBuilder(FieldVector vector) {
        System.out.println("TensorBuilder with vector");
        values = vector;
    }

    public TensorBuilder(IPCClient client, Collection<Integer> shape) {
        System.out.println("TensorBuilder with shape");
        this.shape = shape;
        int count = 0;
        for (Integer item : shape) {
            count += item;
        }
        //TODO:Support other type.
        size = count * Integer.SIZE;
        //TODO: init intBuilder
    }

    public TensorBuilder(IPCClient client, Collection<Integer> shape, FieldVector values) {
        System.out.println("TensorBuilder with shape");
        this.shape = shape;
        int count = 0;
        for (Integer item : shape) {
            count += item.intValue();
        }
        System.out.print("count:" + count);
        //TODO:Support other type.
        size = count * Integer.SIZE;
        try {
            bufferBuilder = new BufferBuilder(client, size);
            for (int i = 0; i < values.getValueCount(); i++) {
                bufferBuilder.getBuffer().setInt(i, ((IntVector)values).get(i));
            }
        } catch (VineyardException e) {
            System.out.println("TensorBuilder error");
            e.printStackTrace();
        }
        System.out.println("TensorBuilder end");
    }

    public void setValues(FieldVector values) {
        System.out.println("setValues ");
        this.values = values;
        // other type
        this.size = values.getBufferSize() * Integer.SIZE;
    }

    public FieldVector getBuffer() {
        return values;
    }

    public int getRowCount() {
        System.out.println("getRowCount");
        System.out.println("Row count " + values.getValueCount());
        return values.getValueCount();
    }

    @Override
    public void build(Client client) {}

    @Override
    public ObjectMeta seal(Client client) {
        System.out.println("seal tensor");
        this.build(client);
        ObjectMeta tensorMeta = ObjectMeta.empty();

        // TODO: set other typename
        tensorMeta.setTypename("vineyard::Tensor<int>");
        tensorMeta.setValue("value_type_", "int");
        // Int32ArrayBuilder intBuilder = new Int32ArrayBuilder(client, );

        try {
            tensorMeta.addMember("buffer_", bufferBuilder.seal(client));
        } catch (VineyardException e) {
            System.out.println("Seal buffer error");
            e.printStackTrace();
            return null;
        }
        // tensorMeta.addMember("buffer_", ????);
        tensorMeta.setListValue("shape_", shape);
        tensorMeta.setValue("partition_index_", partition_index_);
        tensorMeta.setNBytes(size);

        try {
            client.createMetaData(tensorMeta);
            System.out.println("Tensor id:" + tensorMeta.getId().value());
            System.out.println("Tensor id:" + tensorMeta.getId());
        } catch (VineyardException e) {
            e.printStackTrace();
            return null;
        }
        return tensorMeta;
    }
}
