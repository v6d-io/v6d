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
package io.v6d.hive.ql.io;

import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.RecordBatchBuilder;
import io.v6d.modules.basic.arrow.SchemaBuilder;
import io.v6d.modules.basic.arrow.TableBuilder;
import lombok.val;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.NonNullableStructVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hadoop.hive.ql.exec.vector.*;
import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
// how to connect vineyard

// check phase:
// 1. check if vineyard is running
// 2. check if file exist in vineyard
// 3. if exist, read file from vineyard
// 4. if not exist, read file from hdfs and write to vineyard
//
// We do not split the file at present.

public class VineyardArrowInputFormat extends HiveInputFormat<NullWritable, ArrowWrapperWritable> implements VectorizedInputFormatInterface {

    @Override
    public RecordReader<NullWritable, ArrowWrapperWritable>
    getRecordReader(InputSplit genericSplit, JobConf job, Reporter reporter)
            throws IOException {
        reporter.setStatus(genericSplit.toString());
        System.out.printf("--------+creating vineyard record reader\n");
        System.out.println("split class:" + genericSplit.getClass().getName());
        return new VineyardArrowRecordReader(job, (VineyardSplit) genericSplit);
    }

    @Override
    public VectorizedSupport.Support[] getSupportedFeatures() {
        return new VectorizedSupport.Support[] {VectorizedSupport.Support.DECIMAL_64};
    }

    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
        System.out.println("--------+creating vineyard input split. Num:" + numSplits);
        List<InputSplit> splits = new ArrayList<InputSplit>();
        Path path = FileInputFormat.getInputPaths(job)[0];
        System.out.println("path:" + path);
        // fill splits
        VineyardSplit vineyardSplit = new VineyardSplit();
        vineyardSplit.setPath(path);
        splits.add(vineyardSplit);
        return splits.toArray(new VineyardSplit[splits.size()]);
    }
}

class VineyardArrowRecordReader implements RecordReader<NullWritable, ArrowWrapperWritable> {
    private static Logger logger = LoggerFactory.getLogger(VineyardArrowRecordReader.class);

    // vineyard field
    private static IPCClient client;
    private String tableName;
    private Boolean tableNameValid = false;
    private VectorSchemaRoot vectorSchemaRoot;

    private VectorizedRowBatchCtx ctx;

    // for test
    private long tableID;
    private TableBuilder tableBuilder;
    private SchemaBuilder schemaBuilder;
    private List<RecordBatchBuilder> recordBatchBuilders;
    private static RecordBatchBuilder recordBatchBuilder;

    VineyardArrowRecordReader(JobConf job, VineyardSplit split) {
        System.out.printf("--------+creating vineyard record reader\n");
        // throw new RuntimeException("mapred record reader: unimplemented");
        // reader = new LineRecordReader(job, split);
        System.out.println("Path:" + split.getPath());
        String path = split.getPath().toString();
        int index = path.lastIndexOf("/");
        tableName = path.substring(index + 1);
        tableNameValid = true;
        System.out.println("Table name:" + tableName);

        // connect to vineyard
        if (client == null) {
            // TBD: get vineyard socket path from table properties
            try {
                client = new IPCClient("/tmp/vineyard.sock");
            } catch (Exception e) {
                System.out.println("connect to vineyard failed!");
                System.out.println(e.getMessage());
            }
        }
        if (client == null || !client.connected()) {
            System.out.println("connected to vineyard failed!");
            return;
        } else {
            System.out.println("connected to vineyard succeed!");
            System.out.println("Hello, vineyard!");
        }

        try {
            prepare();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        Arrow.instantiate();
        ctx = Utilities.getVectorizedRowBatchCtx(job);
    }

    @Override
    public void close() throws IOException {
        System.out.printf("--------closing\n");
        if(client != null && client.connected()) {
            client.disconnect();
            System.out.println("Bye, vineyard!");
        }
    }

    @Override
    public NullWritable createKey() {
        System.out.printf("--------creating key\n");
        return null;
    }

    @Override
    public ArrowWrapperWritable createValue() {
        System.out.printf("++++++++creating value\n");
        // return ctx.createVectorizedRowBatch();
        return new ArrowWrapperWritable(null , Arrow.default_allocator, NonNullableStructVector.empty(tableName, Arrow.default_allocator));
    }

    @Override
    public long getPos() throws IOException {
        System.out.printf("+++++++get pos\n");
        return 0;
    }

    @Override
    public float getProgress() throws IOException {
        System.out.printf("++++++++get progress\n");
        return 0;
    }

    private void arrowToVectorizedRowBatch(VectorSchemaRoot recordBatch, VectorizedRowBatch batch) {
        batch.numCols = recordBatch.getFieldVectors().size();
        batch.size = recordBatch.getRowCount();
        batch.selected = new int[batch.size];
        batch.selectedInUse = true;
        batch.cols = new ColumnVector[batch.numCols];
        batch.projectedColumns = new int[batch.numCols];
        batch.projectionSize = batch.numCols;
        for (int i = 0; i < batch.numCols; i++) {
            batch.projectedColumns[i] = i;
        }
    }

    @Override
    public boolean next(NullWritable key, ArrowWrapperWritable value) throws IOException {
        System.out.printf("+++++++++next\n");
        if (tableNameValid) {
            // read(tableName);
            // if exist, create vectorSchema root set data and return true;
            // return false;
            vectorSchemaRoot = getSchemaRoot();
            if (vectorSchemaRoot == null) {
                System.out.println(tableName + " not exist!");
                return false;
            }
            System.out.println("Get schema root succeed!");
            value.setVectorSchemaRoot(vectorSchemaRoot);

            // this.arrowToVectorizedRowBatch(vectorSchemaRoot, value);

            // value.setVectorSchemaRoot(vectorSchemaRoot);
            // NonNullableStructVector vector = value.getRootVector();
            // vector.initializeChildrenFromFields(vectorSchemaRoot.getSchema().getFields());
            // vector.setValueCount(vectorSchemaRoot.getRowCount());
            // List<FieldVector> fieldVectors = vector.getChildrenFromFields();
            // System.out.println("fieldVectors size: " + fieldVectors.size());
            // for (int i = 0; i < fieldVectors.size(); i++) {
            //     System.out.println("fieldVectors[" + i + "] size: " + fieldVectors.get(i).getValueCount());
            //     for (int j = 0; j < fieldVectors.get(i).getValueCount(); j++) {
            //         ((org.apache.arrow.vector.IntVector)(fieldVectors.get(i))).set(j, (int)(vectorSchemaRoot.getFieldVectors().get(i).getObject(j)));
            //         System.out.println(fieldVectors.get(i).getObject(j));
            //
            //     }
            //     System.out.println("null count:" + ((org.apache.arrow.vector.IntVector)fieldVectors.get(i)).getNullCount());
            // }
            System.out.println("========================");
            for (int i = 0; i < vectorSchemaRoot.getSchema().getFields().size(); i++) {
                for (int j = 0; j < vectorSchemaRoot.getRowCount(); j++) {
                    System.out.println(vectorSchemaRoot.getFieldVectors().get(i).getObject(j));
                }
                System.out.println("null count:" + ((org.apache.arrow.vector.IntVector)vectorSchemaRoot.getFieldVectors().get(i)).getNullCount());
            }
            // if (value.getVectorSchemaRoot() != null) {
            //     System.out.println("Set vectorSchemaRoot succeed!");
            // }
            return true;
        }
        return false;
    }

    private VectorSchemaRoot getSchemaRoot() {
        // System.out.println("Get table from vineyard.");
        // if (tableNameValid == false) {
        //     return null;
        // }

        // try {
        //     ObjectID objectID = client.getName(tableName);
        //     System.out.println("ObjectID: " + objectID.value());
        //     ObjectMeta meta = client.getMetaData(objectID);
        //     if (meta == null) {
        //         System.out.println("Get meta failed.");
        //     } else {
        //         System.out.println("Get meta succeed. Meta id:" + meta.getId().value());
        //     }
        //     Table table = (Table) ObjectFactory.getFactory().resolve(meta);
        //     if (table == null) {
        //         System.out.println("Create table from factory failed!");
        //     } else {
        //         System.out.println("Create table from factory succeed!");
        //     }
        //     // TBD: support more than one batch
        //     tableNameValid = false;
        //     return table.getArrowBatch(0);
        // } catch (Exception e) {
        //     System.out.println("Get table failed!");
        //     System.out.println(e.getMessage());
        // }
        BufferAllocator allocator = new RootAllocator();
        Field field1 = new Field("field_1", FieldType.nullable(new ArrowType.Int(32, true)), null);//Arrow.makeField("field_1", Arrow.FieldType.Int);
        Field field2 = new Field("field_2", FieldType.nullable(new ArrowType.Int(32, true)), null);//Arrow.makeField("field_2", Arrow.FieldType.Int);
        List<Field> fields = new ArrayList<Field>();
        fields.add(field1);
        fields.add(field2);
        // org.apache.arrow.vector.IntVector vector1 = new org.apache.arrow.vector.IntVector(field1, Arrow.default_allocator);
        // org.apache.arrow.vector.IntVector vector2 = new org.apache.arrow.vector.IntVector(field2, Arrow.default_allocator);
        // vector1.allocateNew(5);
        // vector2.allocateNew(5);
        // for (int i = 0; i < 5; i++) {
        //     vector1.set(i, i);
        //     vector2.set(i, i + 1);
        // }
        // List<FieldVector> fieldVectors = new ArrayList<FieldVector>();
        // fieldVectors.add(vector1);
        // fieldVectors.add(vector2);

        // VectorSchemaRoot vectorSchemaRoot = new VectorSchemaRoot(fieldVectors);
        Schema schema = new Schema(fields);
        VectorSchemaRoot vectorSchemaRoot = VectorSchemaRoot.create(schema, allocator);
        vectorSchemaRoot.setRowCount(5);
        org.apache.arrow.vector.IntVector v1 = (org.apache.arrow.vector.IntVector) vectorSchemaRoot.getVector("field_1");
        org.apache.arrow.vector.IntVector v2 = (org.apache.arrow.vector.IntVector) vectorSchemaRoot.getVector("field_2");
        v1.allocateNew(5);
        v1.setValueCount(5);
        v2.allocateNew(5);
        v2.setValueCount(5);
        for (int i = 0; i < 5; i++) {
            v1.set(i, i);
            v2.set(i, i + 1);
        }
        tableNameValid = false;
        return vectorSchemaRoot;
    }

    private void prepare() throws VineyardException{
        System.out.println("Prepare data");
        SchemaBuilder schemaBuilder = new SchemaBuilder();
        recordBatchBuilders = new ArrayList<RecordBatchBuilder>();
        Field field1 = Arrow.makeField("field_1", Arrow.FieldType.Int);
        Field field2 = Arrow.makeField("field_2", Arrow.FieldType.Int);
        schemaBuilder.addField(field1);
        schemaBuilder.addField(field2);
        recordBatchBuilder = new RecordBatchBuilder(client, 5);
        recordBatchBuilder.addField(field1);
        recordBatchBuilder.addField(field2);
        recordBatchBuilder.finishSchema(client);

        val column1 = recordBatchBuilder.getColumnBuilder(0);
        column1.setInt(0, 1);
        column1.setInt(1, 2);
        column1.setInt(2, 3);
        column1.setInt(3, 4);
        column1.setInt(4, 5);

        val column2 = recordBatchBuilder.getColumnBuilder(1);
        column2.setInt(0, 2);
        column2.setInt(1, 3);
        column2.setInt(2, 4);
        column2.setInt(3, 5);
        column2.setInt(4, 6);

        recordBatchBuilders.add(recordBatchBuilder);
        tableBuilder = new TableBuilder(client, schemaBuilder, recordBatchBuilders);
        ObjectMeta meta = tableBuilder.seal(client);
        client.persist(meta.getId());
        client.putName(meta.getId(), "hive_example");
        System.out.println("Table ID: " + meta.getId().value());
        System.out.println("Prepare data done");
    }
}
// how to get data from vineyard ?????