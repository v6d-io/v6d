package io.v6d.modules.basic.dataframe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.v6d.core.client.Client;
import io.v6d.core.client.ds.ObjectBase;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.TableBuilder;
import io.v6d.modules.basic.tensor.TensorBuilder;
import io.v6d.modules.basic.tensor.Tensor;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataFrameBuilder implements ObjectBuilder {

    private Logger logger = LoggerFactory.getLogger(TableBuilder.class);
    private Map<JsonNode, TensorBuilder> values;
    private List<JsonNode> columns;
    private List<TensorBuilder> tensorBuilders;
    private int rowCount;
    private int columnCount;
    private ObjectMapper mapper;

    private JsonNode getJsonNode(String context) {
        JsonNode newNode;
        try {
            newNode = mapper.valueToTree(context);
        } catch (Exception e) {
            System.out.println("Exception " + e);
            return null;
        }
        return newNode;
    }

    public DataFrameBuilder(Client client) {
        values = new java.util.HashMap<>();
        columnCount = 0;
        rowCount = 0;
        mapper = new ObjectMapper();
        columns = new ArrayList<>();
        tensorBuilders = new ArrayList<>();
        System.out.println("values size:" + values.size());
    }

    public void set_index(TensorBuilder index) {
        values.put(getJsonNode("index_"), index);
        rowCount = index.getRowCount();
        columns.add(getJsonNode("index_"));
        tensorBuilders.add(index);
        columnCount++;
    }

    public TensorBuilder column(String column) {
        return values.get(getJsonNode(column));
    }

    public void addColumn(String column, TensorBuilder builder) {
        System.out.println("AddColumn " + column);
        values.put(getJsonNode(column), builder);
        columnCount++;
        columns.add(getJsonNode(column));
        tensorBuilders.add(builder);
        System.out.println("values size:" + values.size());
    }

    @Override
    public void build(Client client) {
        // TBD
        System.out.println("build");
    }

    // In C++, Seal use the object created by builder as return value.
    // But here return objectMeta. Maybe we should return object in the future.
    @Override
    public ObjectMeta seal(Client client) {
        // TBD
        System.out.println("seal dataframe");
        this.build(client);
        ObjectMeta dataFrameMeta = ObjectMeta.empty();
        
        // seal tensor
        // for (int i = 0; i < columnCount; i++) {
        //     // tensors.add(new Tensor(tensorMeta, null, null, null, null))
        //     tensorBuilders.get(i).seal(client);
        // }
        
        //seal dataframe
        dataFrameMeta.setTypename("vineyard::DataFrame");
        
        // TODO: support in the future ?
        /*dataFrameMeta.setValue("partition_index_row_", 0);
        dataFrameMeta.setValue("partition_index_column_", 0);
        dataFrameMeta.setValue("row_batch_index_", 0);*/

        dataFrameMeta.setListValue("columns_", columns);
        dataFrameMeta.setValue("__values_-size", values.size());
        System.out.println("values size:" + values.size());
        // seal tensor
        values.forEach((key,value)->{
            System.out.println("key:" + key.toString());
            dataFrameMeta.setValue("__values_-key-", key);
            dataFrameMeta.addMember("__values_-value-", value.seal(client));
        });

        // DataFrame dataFrame = new DataFrame(dataFrameMeta, rowCount, columnCount, columns, tensors);
        // dataFrame.setColumns(columns);
        // dataFrame.setColumnCount(columnCount);
        // dataFrame.setRowCount(rowCount);
        // dataFrame.setValues(tensors);
        try {
            return client.createMetaData(dataFrameMeta);
        } catch (VineyardException e) {
            System.out.println("Exception " + e);
            return null;
        }
    }
}
