package io.v6d.modules.basic.tensor;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBase;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.TableBuilder;

import java.util.ArrayList;

import org.apache.arrow.vector.FieldVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TensorBuilder implements ObjectBuilder{
    private Logger logger = LoggerFactory.getLogger(TableBuilder.class);
    private FieldVector values;
    private Client client;

    public TensorBuilder(Client client, FieldVector vector) {
        this.client = client;
        values = vector;
    }

    public void append(Object value) {
        System.out.println("append " + value);
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

        // TODO: set the typename
        tensorMeta.setTypename("vineyard::Tensor<int>");
        try {
            client.createMetaData(tensorMeta);
        } catch (VineyardException e) {
            e.printStackTrace();
            return null;
        }
        return tensorMeta;
    }
}
