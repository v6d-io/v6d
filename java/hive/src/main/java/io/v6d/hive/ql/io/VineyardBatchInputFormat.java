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

import org.apache.arrow.memory.BufferAllocator;

import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.TableBuilder;
import io.v6d.modules.basic.arrow.SchemaBuilder;
import io.v6d.modules.basic.arrow.Table;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.RecordBatchBuilder;

import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hadoop.hive.ql.exec.vector.*;
import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.Reporter;

import org.apache.arrow.vector.*;
import org.apache.arrow.vector.util.Text;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.hadoop.fs.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.io.IOException;

import lombok.val;

public class VineyardBatchInputFormat extends HiveInputFormat<NullWritable, VectorizedRowBatch> implements VectorizedInputFormatInterface {

    @Override
    public RecordReader<NullWritable, VectorizedRowBatch>
    getRecordReader(InputSplit genericSplit, JobConf job, Reporter reporter)
            throws IOException {
        reporter.setStatus(genericSplit.toString());
        return new VineyardBatchRecordReader(job, (VineyardSplit) genericSplit);
    }

    @Override
    public VectorizedSupport.Support[] getSupportedFeatures() {
        return new VectorizedSupport.Support[] {VectorizedSupport.Support.DECIMAL_64};
    }

    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
        List<InputSplit> splits = new ArrayList<InputSplit>();
        Path path = FileInputFormat.getInputPaths(job)[0];
        // fill splits
        VineyardSplit vineyardSplit = new VineyardSplit(path, 0, 0, job);
        splits.add(vineyardSplit);
        return splits.toArray(new VineyardSplit[splits.size()]);
    }
}

class VineyardBatchRecordReader implements RecordReader<NullWritable, VectorizedRowBatch> {
    private static Logger logger = LoggerFactory.getLogger(VineyardBatchRecordReader.class);

    // vineyard field
    private static IPCClient client;
    private String tableName;
    private Boolean tableNameValid = false;
    private VectorSchemaRoot vectorSchemaRoot;

    private VectorizedRowBatchCtx ctx;

    // for test
    private long tableID;
    private Table table;
    private TableBuilder tableBuilder;
    private SchemaBuilder schemaBuilder;
    private List<RecordBatchBuilder> recordBatchBuilders;
    private static RecordBatchBuilder recordBatchBuilder;
    private int batchIndex = 0;
    private int batchNum = 0;

    VineyardBatchRecordReader(JobConf job, VineyardSplit split) {
        String path = split.getPath().toString();
        int index = path.lastIndexOf("/");
        tableName = path.substring(index + 1);
        tableNameValid = true;

        // connect to vineyard
        if (client == null) {
            // TBD: get vineyard socket path from table properties
            try {
                client = new IPCClient("/tmp/vineyard/vineyard.sock");
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

        Arrow.instantiate();
        ctx = Utilities.getVectorizedRowBatchCtx(job);
    }

    @Override
    public void close() throws IOException {
        if(client != null && client.connected()) {
            client.disconnect();
            System.out.println("Bye, vineyard!");
        }
    }

    @Override
    public NullWritable createKey() {
        return null;
    }

    @Override
    public VectorizedRowBatch createValue() {
        return ctx.createVectorizedRowBatch();
    }

    @Override
    public long getPos() throws IOException {
        return 0;
    }

    @Override
    public float getProgress() throws IOException {
        return batchIndex / batchNum;
    }

    /**
     * Copy data from the Arrow RecordBatch to the VectorizedRowBatch.
     *
     * @param recordBatch the Arrow RecordBatch to copy from
     * @param batch the VectorizedRowBatch to copy to
     */
    private void arrowToVectorizedRowBatch(VectorSchemaRoot recordBatch, VectorizedRowBatch batch) {
        // batch.numCols = recordBatch.getFieldVectors().size();
        batch.size = recordBatch.getRowCount();
        batch.selected = new int[batch.size];
        batch.selectedInUse = false;
        // batch.cols = new ColumnVector[batch.numCols];
        // batch.projectedColumns = new int[batch.numCols];
        // batch.projectionSize = batch.numCols;
        // for (int i = 0; i < batch.numCols; i++) {
        //     batch.projectedColumns[i] = i;
        // }

        for (int i = 0; i < recordBatch.getSchema().getFields().size(); i++) {
            Field field = recordBatch.getSchema().getFields().get(i);
            if (field.getType().equals(Arrow.Type.Boolean)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BitVector bitVector = (BitVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bitVector.get(k);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Int)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                IntVector intVector = (IntVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = intVector.get(k);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Int64)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BigIntVector bigIntVector = (BigIntVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bigIntVector.get(k);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Float)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float4Vector float4Vector = (Float4Vector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = float4Vector.get(k);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Double)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float8Vector float8Vector = (Float8Vector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = float8Vector.get(k);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.LargeVarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                LargeVarCharVector largeVarCharVector = (LargeVarCharVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    vector.setRef(k, largeVarCharVector.get(k), 0, largeVarCharVector.get(k).length);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.VarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                vector.init();
                VarCharVector varCharVector = (VarCharVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    vector.setRef(k, varCharVector.get(k), 0, varCharVector.get(k).length);
                }
                batch.cols[i] = vector;
            } else {
                System.out.println("array builder for type " + field.getType() + " is not supported");
                throw new UnsupportedOperationException("array builder for type " + field.getType() + " is not supported");
            }
        }
    }

    @Override
    public boolean next(NullWritable key, VectorizedRowBatch value) throws IOException {
        if (tableNameValid) {
            // read(tableName);
            // if exist, create vectorSchema root set data and return true;
            // return false;

            // vectorSchemaRoot = getSchemaRoot();
            
            //test vineyard
            if (table == null) {
                try {
                    // long id = tableID;//client.getName(tableName);
                    // tableID = 
                    ObjectID objectID = client.getName(tableName, false);
                    table = (Table) ObjectFactory.getFactory().resolve(client.getMetaData(objectID));
                    batchNum = table.getBatches().size();
                } catch (Exception e) {
                    System.out.println("Get objectID failed.");
                    return false;
                }
            }
            if (batchIndex >= table.getBatches().size()) {
                return false;
            }
            vectorSchemaRoot = table.getArrowBatch(batchIndex++);

            if (vectorSchemaRoot == null) {
                System.out.println(tableName + " not exist!");
                return false;
            }

            this.arrowToVectorizedRowBatch(vectorSchemaRoot, value);
            return true;
        }
        return false;
    }

    // For test.
    private VectorSchemaRoot getSchemaRoot() {
        int BATCH_SIZE = 3;

        BufferAllocator allocator = new RootAllocator();
        Field field1 = new Field("field_1", FieldType.nullable(new ArrowType.Utf8()), null);
        Field field2 = new Field("field_2", FieldType.nullable(new ArrowType.LargeUtf8()), null);
        List<Field> fields = new ArrayList<Field>();
        fields.add(field1);
        fields.add(field2);

        Schema schema = new Schema(fields);
        VectorSchemaRoot vectorSchemaRoot = VectorSchemaRoot.create(schema, allocator);
        vectorSchemaRoot.setRowCount(BATCH_SIZE);
        org.apache.arrow.vector.VarCharVector v1 = (VarCharVector) vectorSchemaRoot.getVector("field_1");
        org.apache.arrow.vector.LargeVarCharVector v2 = (LargeVarCharVector) vectorSchemaRoot.getVector("field_2");
        v1.allocateNew(BATCH_SIZE);
        v1.setValueCount(BATCH_SIZE);
        v2.allocateNew(BATCH_SIZE);
        v2.setValueCount(BATCH_SIZE);

        v1.setSafe(0, new Text("我是"));
        v1.setSafe(1, new Text("涛老师"));
        v1.setSafe(2, new Text("的迷弟"));
        v2.setSafe(0, new Text("涛老师"));
        v2.setSafe(1, new Text("太强啦"));
        v2.setSafe(2, new Text("！！！"));
        tableNameValid = false;
        return vectorSchemaRoot;
    }

    // For test.
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
        tableID = meta.getId().value();
        System.out.println("Table ID: " + tableID);
        System.out.println("Prepare data done");
    }
}
// how to get data from vineyard ?????