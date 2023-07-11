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
import org.apache.arrow.memory.RootAllocator;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.TableBuilder;
import io.v6d.modules.basic.arrow.SchemaBuilder;
import io.v6d.modules.basic.arrow.Table;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.RecordBatchBuilder;

import org.apache.arrow.vector.complex.NonNullableStructVector;
import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hadoop.hive.ql.exec.vector.*;
import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.FileSplit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.arrow.vector.util.Text;

import com.google.flatbuffers.LongVector;

import org.apache.hadoop.hive.ql.exec.vector.*;

import java.io.IOException;

import org.apache.arrow.vector.*;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.hadoop.fs.Path;

import java.util.ArrayList;
import java.util.List;
import java.io.DataOutput;
import java.io.DataInput;
import java.util.Vector;

import lombok.val;
// how to connect vineyard

// check phase:
// 1. check if vineyard is running
// 2. check if file exist in vineyard
// 3. if exist, read file from vineyard
// 4. if not exist, read file from hdfs and write to vineyard
//
// We do not split the file at present.

public class VineyardBatchInputFormat extends HiveInputFormat<NullWritable, VectorizedRowBatch> implements VectorizedInputFormatInterface {

    @Override
    public RecordReader<NullWritable, VectorizedRowBatch>
    getRecordReader(InputSplit genericSplit, JobConf job, Reporter reporter)
            throws IOException {
        reporter.setStatus(genericSplit.toString());
        System.out.printf("--------+creating vineyard record reader\n");
        System.out.println("split class:" + genericSplit.getClass().getName());
        return new VineyardBatchRecordReader(job, (VineyardSplit) genericSplit);
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

    VineyardBatchRecordReader(JobConf job, VineyardSplit split) {
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
    public VectorizedRowBatch createValue() {
        System.out.printf("++++++++creating value\n");
        return ctx.createVectorizedRowBatch();
        // return new ArrowWrapperWritable(null , Arrow.default_allocator, NonNullableStructVector.empty(tableName, Arrow.default_allocator));
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
        batch.selectedInUse = false;
        batch.cols = new ColumnVector[batch.numCols];
        batch.projectedColumns = new int[batch.numCols];
        batch.projectionSize = batch.numCols;
        for (int i = 0; i < batch.numCols; i++) {
            batch.projectedColumns[i] = i;
        }

        for (int i = 0; i < recordBatch.getSchema().getFields().size(); i++) {
            Field field = recordBatch.getSchema().getFields().get(i);
            if (field.getType().equals(Arrow.Type.Boolean)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BitVector bitVector = (BitVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bitVector.get(k);
                }
                batch.cols[i] = vector;
                for (int k = 0; k < batch.size; k++) {
                    System.out.printf(vector.vector[k] + " ");
                }
                System.out.printf("\n");
            } else if (field.getType().equals(Arrow.Type.Int)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                IntVector intVector = (IntVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = intVector.get(k);
                }
                batch.cols[i] = vector;
                for (int k = 0; k < batch.size; k++) {
                    System.out.printf(vector.vector[k] + " ");
                }
                System.out.printf("\n");
            } else if (field.getType().equals(Arrow.Type.Int64)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BigIntVector bigIntVector = (BigIntVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bigIntVector.get(k);
                }
                batch.cols[i] = vector;
                for (int k = 0; k < batch.size; k++) {
                    System.out.printf(vector.vector[k] + " ");
                }
                System.out.printf("\n");
            } else if (field.getType().equals(Arrow.Type.Float)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float4Vector float4Vector = (Float4Vector) recordBatch.getFieldVectors().get(i);
                System.out.println("Batch size is :" + batch.size);
                for (int k = 0; k < batch.size; k++) {
                    System.out.println("k is " + k + " value is " + float4Vector.get(k));
                    vector.vector[k] = float4Vector.get(k);
                }
                batch.cols[i] = vector;
                for (int k = 0; k < batch.size; k++) {
                    System.out.printf(vector.vector[k] + " ");
                }
                System.out.printf("\n");
            } else if (field.getType().equals(Arrow.Type.Double)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float8Vector float8Vector = (Float8Vector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = float8Vector.get(k);
                }
                batch.cols[i] = vector;
                for (int k = 0; k < batch.size; k++) {
                    System.out.printf(vector.vector[k] + " ");
                }
                System.out.printf("\n");
            } else if (field.getType().equals(Arrow.Type.LargeVarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                LargeVarCharVector largeVarCharVector = (LargeVarCharVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    // vector.vector[k] = new byte[largeVarCharVector.getByteCapacity()];
                    System.out.println("byte capacity:" + largeVarCharVector.getByteCapacity());
                    System.out.println("buffer size:" + largeVarCharVector.get(k).length);
                    
                    vector.setRef(k, largeVarCharVector.get(k), 0, largeVarCharVector.get(k).length);
                    System.out.println("vector size:" + vector.vector[k].length);
                }
                System.out.println("Large");
                for (int p = 0; p < batch.size; p++) {
                    System.out.println((vector.toString(p)) + ".");
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.VarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                vector.init();
                VarCharVector varCharVector = (VarCharVector) recordBatch.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    // vector.vector[k] = new byte[(int)(varCharVector.getByteCapacity())];
                    System.out.println("byte capacity:" + varCharVector.getByteCapacity());
                    System.out.println("buffer size:" + varCharVector.get(k).length);
                    vector.setRef(k, varCharVector.get(k), 0, varCharVector.get(k).length);
                    System.out.println("vector size:" + vector.vector[k].length);
                }
                for (int p = 0; p < batch.size; p++) {
                    System.out.println((vector.toString(p)) + ".");
                }
                batch.cols[i] = vector;
            } else {
                // throw new VineyardException.NotImplemented(
                //         "array builder for type " + field.getType() + " is not supported");
                System.out.println("array builder for type " + field.getType() + " is not supported");
            }
        }
        System.out.println("batch size:" + batch.size);
    }

    @Override
    public boolean next(NullWritable key, VectorizedRowBatch value) throws IOException {
        System.out.printf("+++++++++next\n");
        if (tableNameValid) {
            // read(tableName);
            // if exist, create vectorSchema root set data and return true;
            // return false;

            // vectorSchemaRoot = getSchemaRoot();
            
            //test vineyard
            if (table == null) {
                try {
                    long id = tableID;//client.getName(tableName);
                    ObjectID objectID = new ObjectID(id);
                    table = (Table) ObjectFactory.getFactory().resolve(client.getMetaData(objectID));
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
            System.out.println("Get schema root succeed!");
            // value.setVectorSchemaRoot(vectorSchemaRoot);

            this.arrowToVectorizedRowBatch(vectorSchemaRoot, value);
            return true;
        }
        return false;
    }

    private VectorSchemaRoot getSchemaRoot() {
        int BATCH_SIZE = 3;
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
        // Field field1 = new Field("field_1", FieldType.nullable(new ArrowType.Int(32, true)), null);//Arrow.makeField("field_1", Arrow.FieldType.Int);
        // Field field2 = new Field("field_2", FieldType.nullable(new ArrowType.Int(32, true)), null);//Arrow.makeField("field_2", Arrow.FieldType.Int);
        // Field field1 = new Field("field_1", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)), null);
        // Field field2 = new Field("field_2", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null);
        Field field1 = new Field("field_1", FieldType.nullable(new ArrowType.Utf8()), null);
        Field field2 = new Field("field_2", FieldType.nullable(new ArrowType.LargeUtf8()), null);
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
        vectorSchemaRoot.setRowCount(BATCH_SIZE);
        // org.apache.arrow.vector.IntVector v1 = (org.apache.arrow.vector.IntVector) vectorSchemaRoot.getVector("field_1");
        // org.apache.arrow.vector.IntVector v2 = (org.apache.arrow.vector.IntVector) vectorSchemaRoot.getVector("field_2");
        // org.apache.arrow.vector.Float4Vector v1 = (Float4Vector) vectorSchemaRoot.getVector("field_1");
        // org.apache.arrow.vector.Float8Vector v2 = (Float8Vector) vectorSchemaRoot.getVector("field_2");
        org.apache.arrow.vector.VarCharVector v1 = (VarCharVector) vectorSchemaRoot.getVector("field_1");
        org.apache.arrow.vector.LargeVarCharVector v2 = (LargeVarCharVector) vectorSchemaRoot.getVector("field_2");
        v1.allocateNew(BATCH_SIZE);
        v1.setValueCount(BATCH_SIZE);
        v2.allocateNew(BATCH_SIZE);
        v2.setValueCount(BATCH_SIZE);
        // for (int i = 0; i < 3; i++) {
        //     v1.set(i, i);
        //     v2.set(i, i + 1);
        // }
        v1.setSafe(0, new Text("我是"));
        v1.setSafe(1, new Text("涛老师"));
        v1.setSafe(2, new Text("的迷弟"));
        v2.setSafe(0, new Text("涛老师"));
        v2.setSafe(1, new Text("太强啦"));
        v2.setSafe(2, new Text("！！！"));
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
        tableID = meta.getId().value();
        System.out.println("Table ID: " + tableID);
        System.out.println("Prepare data done");
    }
}
// how to get data from vineyard ?????