package io.v6d.modules.basic.dataframe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.v6d.core.client.Client;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.tensor.TensorBuilder;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class DataFrameBuilder implements ObjectBuilder {
    private Map<JsonNode, TensorBuilder> values;
    private List<JsonNode> columns;
    private List<TensorBuilder> tensorBuilders;
    private int rowCount;
    private int columnCount;
    private ObjectMapper mapper;

    private long valuesIndex = 0L;
    private long nBytes = 0L;

    private JsonNode getJsonNode(String context) throws VineyardException {
        JsonNode newNode;
        newNode = mapper.valueToTree(context);
        return newNode;
    }

    public DataFrameBuilder(Client client) throws VineyardException {
        values = new java.util.HashMap<>();
        columnCount = 0;
        rowCount = 0;
        mapper = new ObjectMapper();
        columns = new ArrayList<>();
        tensorBuilders = new ArrayList<>();
        if (tensorBuilders == null || columns == null || mapper == null) {
            throw new VineyardException.NotEnoughMemory("Failed to create tensorBuilders");
        }
    }

    public void set_index(TensorBuilder index) throws VineyardException {
        values.put(getJsonNode("index_"), index);
        rowCount = index.getRowCount();
        columns.add(getJsonNode("index_"));
        tensorBuilders.add(index);
        columnCount++;
    }

    public TensorBuilder column(String column) throws VineyardException {
        return values.get(getJsonNode(column));
    }

    public void addColumn(String column, TensorBuilder builder) throws VineyardException {
        System.out.println("addColumn " + column);
        values.put(getJsonNode(column), builder);
        columnCount++;
        columns.add(getJsonNode(column));
        tensorBuilders.add(builder);
    }

    @Override
    public void build(Client client) throws VineyardException {}

    // In C++, Seal use the object created by builder as return value.
    // But here return objectMeta. Maybe we should return object in the future.
    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);
        ObjectMeta dataFrameMeta = ObjectMeta.empty();
        
        //seal dataframe
        dataFrameMeta.setTypename("vineyard::DataFrame");
        
        //support in the future.
        dataFrameMeta.setValue("partition_index_row_", 0);
        dataFrameMeta.setValue("partition_index_column_", 0);
        dataFrameMeta.setValue("row_batch_index_", 0);

        dataFrameMeta.setListValue("columns_", columns);
        dataFrameMeta.setValue("__values_-size", values.size());
        // seal tensor

        values.forEach((key,value)->{
            System.out.println("key:" + key.toString());
            dataFrameMeta.setValue("__values_-key-" + String.valueOf(valuesIndex), key);
            try {
                dataFrameMeta.addMember("__values_-value-" + String.valueOf(valuesIndex), value.seal(client));
            } catch (VineyardException e) {
                System.out.println("Exception " + e.getMessage());
            }
            nBytes += value.getNBytes();
            valuesIndex++;
        });
        dataFrameMeta.setNBytes(nBytes);

        return client.createMetaData(dataFrameMeta);
    }
}
