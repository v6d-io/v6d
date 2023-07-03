package io.v6d.modules.basic.tensor;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.Buffer;
import io.v6d.modules.basic.arrow.BufferBuilder;
import io.v6d.modules.basic.arrow.Int32ArrayBuilder;
import io.v6d.core.common.util.VineyardException;

import java.util.Collection;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.Float4Vector;

public class TensorBuilder implements ObjectBuilder{
    private FieldVector values;
    private BufferBuilder bufferBuilder;
    private Collection<Integer> shape;
    private String dtype;
    private int partition_index_ = 0;
    private long size = 0;

    public TensorBuilder(FieldVector vector) throws VineyardException {
        values = vector;
    }

    public TensorBuilder(IPCClient client, Collection<Integer> shape) throws VineyardException {
        this.shape = shape;
        int count = 0;
        for (Integer item : shape) {
            count += item;
        }
        //TODO:Support other type.
        size = count * Integer.SIZE;
        bufferBuilder = new BufferBuilder(client, size);
        if (bufferBuilder == null) {
            throw new VineyardException.NotEnoughMemory("Failed to create bufferBuilder");
        }
    }

    public TensorBuilder(IPCClient client, Collection<Integer> shape, FieldVector values) throws VineyardException {
        this.shape = shape;
        int count = 0;
        for (Integer item : shape) {
            count += item.intValue();
        }

        //TODO:Support other type.
        switch (values.getField().getType().getTypeID()) {
            case Int:
                dtype = new String("int");
                size = count * Integer.SIZE;
                bufferBuilder = new BufferBuilder(client, size);
                for (int i = 0; i < values.getValueCount(); i++) {
                    bufferBuilder.getBuffer().setInt(i, ((IntVector)values).get(i));
                }
                break;
            case FloatingPoint:
                dtype = new String("float");
                size = count * Float.SIZE;
                bufferBuilder = new BufferBuilder(client, size);
                for (int i = 0; i < values.getValueCount(); i++) {
                    bufferBuilder.getBuffer().setFloat(i, ((Float4Vector)values).get(i));
                }
                break;
            case Utf8:
                //string
                // support in the future.
            default:
                System.out.println("Type:" + values.getField().getType().getTypeID());
                System.out.println("Type class: " + values.getField().getType().getClass());
                throw new VineyardException.NotImplemented("Type Not implemented");
        }
    }

    public void append(Object value) throws VineyardException {
        System.out.println("append");
        // Support in the future.
    }

    public FieldVector getBuffer() {
        return values;
    }

    public int getRowCount() {
        return values.getValueCount();
    }

    public long getNBytes() {
        return size;
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        ObjectMeta tensorMeta = ObjectMeta.empty();

        // TODO: set other typename
        tensorMeta.setTypename("vineyard::Tensor<" + dtype + ">");
        tensorMeta.setValue("value_type_", dtype);
        // Int32ArrayBuilder intBuilder = new Int32ArrayBuilder(client, );

        ObjectMeta meta = bufferBuilder.seal(client);
        tensorMeta.addMember("buffer_", meta);
        // tensorMeta.addMember("buffer_", ????);
        tensorMeta.setListValue("shape_", shape);
        tensorMeta.setValue("partition_index_", partition_index_);
        tensorMeta.setNBytes(size);


        client.createMetaData(tensorMeta);
        return tensorMeta;
    }
}
