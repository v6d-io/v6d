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

import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.ObjectID;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.Table;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hadoop.hive.ql.exec.vector.*;
import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardVectorizedInputFormat extends HiveInputFormat<NullWritable, VectorizedRowBatch>
        implements VectorizedInputFormatInterface {
    public VineyardVectorizedInputFormat() {
        super();
        Context.println("VineyardInputFormat");
    }

    @Override
    public RecordReader<NullWritable, VectorizedRowBatch> getRecordReader(
            InputSplit genericSplit, JobConf job, Reporter reporter) throws IOException {
        Context.println("getRecordReader");
        reporter.setStatus(genericSplit.toString());
        return new VineyardBatchRecordReader(job, (VineyardSplit) genericSplit);
    }

    // @Override
    // public VectorizedSupport.Support[] getSupportedFeatures() {
    //     return new VectorizedSupport.Support[] {VectorizedSupport.Support.DECIMAL_64};
    // }

    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
        Context.println("getSplits");
        Context.println(
                "Utilities.getPlanPath(conf) in get splits is " + Utilities.getPlanPath(job));
        (new Exception()).printStackTrace();
        List<InputSplit> splits = new ArrayList<InputSplit>();
        Path paths[] = FileInputFormat.getInputPaths(job);

        IPCClient client;
        ObjectID tableObjectID[];
        Table table[];
        try {
            client = new IPCClient(System.getenv("VINEYARD_IPC_SOCKET"));
        } catch (Exception e) {
            Context.println("Connect vineyard failed.");
            return splits.toArray(new VineyardSplit[splits.size()]);
        }
        Arrow.instantiate();

        table = new Table[paths.length];
        tableObjectID = new ObjectID[paths.length];
        for (int i = 0; i < paths.length; i++) {
            // int index = path.toString().lastIndexOf("/");
            // Context.println("Path:" + path.toString());
            // String tableName = path.toString().substring(index + 1);

            // Construct table name.
            Path path = paths[i];
            String tableName = path.toString();
            tableName = tableName.replace("/", "#");
            Context.println("table name:" + tableName);

            // Get table from vineyard.
            try {
                tableObjectID[i] = client.getName(tableName);
                table[i] =
                        (Table)
                                ObjectFactory.getFactory()
                                        .resolve(client.getMetaData(tableObjectID[i]));
            } catch (Exception e) {
                Context.println("Get table failed");
                VineyardSplit vineyardSplit = new VineyardSplit(path, 0, 0, job);
                splits.add(vineyardSplit);
                return splits.toArray(new VineyardSplit[splits.size()]);
            }
        }

        // Split table.
        int batchSize, realNumSplits;
        int totalRecordBatchCount = 0;
        int partitionsSplitCount[] = new int[table.length];

        if (numSplits <= table.length) {
            realNumSplits = table.length;
            for (int i = 0; i < table.length; i++) {
                partitionsSplitCount[i] = 1;
            }
        } else {
            realNumSplits = 0;
            for (int i = 0; i < table.length; i++) {
                totalRecordBatchCount += table[i].getBatches().size();
            }
            batchSize =
                    totalRecordBatchCount / numSplits == 0 ? 1 : totalRecordBatchCount / numSplits;
            for (int i = 0; i < table.length; i++) {
                partitionsSplitCount[i] =
                        (table[i].getBatches().size() + batchSize - 1) / batchSize;
                realNumSplits += partitionsSplitCount[i];
            }
        }

        for (int i = 0; i < partitionsSplitCount.length; i++) {
            int partitionSplitCount = partitionsSplitCount[i];
            int partitionBatchSize = table[i].getBatches().size() / partitionSplitCount;
            Context.println("partitionBatchSize:" + partitionBatchSize);
            for (int j = 0; j < partitionSplitCount; j++) {
                VineyardSplit vineyardSplit =
                        new VineyardSplit(paths[i], 0, "1 3".getBytes().length, job);
                Context.println("split path:" + paths[i].toString());
                if (j == partitionSplitCount - 1) {
                    vineyardSplit.setBatch(
                            j * partitionBatchSize,
                            table[i].getBatches().size() - j * partitionBatchSize);
                } else {
                    vineyardSplit.setBatch(j * partitionBatchSize, partitionBatchSize);
                }
                splits.add(vineyardSplit);
            }
        }
        // int batchSize = table.getBatches().size();
        // int realNumSplits = batchSize < numSplits ? batchSize : numSplits;
        // int size =  table.getBatches().size() / realNumSplits;

        // Fill splits
        // for (int i = 0; i < realNumSplits; i++) {
        //     VineyardSplit vineyardSplit = new VineyardSplit(path, 0, 0, job);
        //     if (i == realNumSplits - 1) {
        //         vineyardSplit.setBatch(i * size, table.getBatches().size() - i * size);
        //     } else {
        //         vineyardSplit.setBatch(i * size, size);
        //     }
        //     splits.add(vineyardSplit);
        // }
        Context.println("num split:" + numSplits + " real num split:" + realNumSplits);
        Context.println("table length:" + table.length);
        for (int i = 0; i < table.length; i++) {
            Context.println("table[" + i + "] batch size:" + table[i].getBatches().size());
            Context.println("table[" + i + "] split count:" + partitionsSplitCount[i]);
        }
        client.disconnect();
        Context.println("Splits size:" + splits.size());
        for (int i = 0; i < splits.size(); i++) {
            Context.println("Split[" + i + "] length:" + splits.get(i).getLength());
        }
        Context.println(
                "Utilities.getPlanPath(conf) in get splits is " + Utilities.getPlanPath(job));
        // HiveConf.setVar(job, HiveConf.ConfVars.PLAN, "vineyard:///");
        return splits.toArray(new VineyardSplit[splits.size()]);
    }
}

class VineyardBatchRecordReader implements RecordReader<NullWritable, VectorizedRowBatch> {
    private static Logger logger = LoggerFactory.getLogger(VineyardBatchRecordReader.class);

    // vineyard field
    private IPCClient client;
    private String tableName;
    private Boolean tableNameValid = false;
    private VectorSchemaRoot vectorSchemaRoot;
    private VectorizedRowBatchCtx ctx;
    private Table table;
    private int recordBatchIndex = 0;
    private int recordBatchInnerIndex = 0;

    private int batchStartIndex;
    private int batchSize;

    private Object[] partitionValues;
    private boolean addPartitionCols = true;

    VineyardBatchRecordReader(JobConf job, VineyardSplit split) throws IOException {
        Context.println("VineyardBatchRecordReader");
        String path = split.getPath().toString();
        // int index = path.lastIndexOf("/");
        // tableName = path.substring(index + 1);
        tableName = path.replace('/', '#');
        Context.println("Table name:" + tableName);
        tableNameValid = true;

        batchStartIndex = split.getBatchStartIndex();
        batchSize = split.getBatchSize();
        recordBatchIndex = batchStartIndex;
        // connect to vineyard
        if (client == null) {
            // TBD: get vineyard socket path from table properties
            try {
                client = new IPCClient(System.getenv("VINEYARD_IPC_SOCKET"));
            } catch (Exception e) {
                Context.println("connect to vineyard failed!");
                Context.println(e.getMessage());
            }
        }
        if (client == null || !client.connected()) {
            Context.println("connected to vineyard failed!");
            return;
        } else {
            Context.println("connected to vineyard succeed!");
            Context.println("Hello, vineyard!");
        }

        Arrow.instantiate();
        // HiveConf.setVar(job, HiveConf.ConfVars.PLAN, "vineyard:///");
        Context.println(
                "flag is :"
                        + HiveConf.getBoolVar(job, HiveConf.ConfVars.HIVE_VECTORIZATION_ENABLED));
        HiveConf.setBoolVar(job, HiveConf.ConfVars.HIVE_VECTORIZATION_ENABLED, true);
        Context.println("Utilities.getPlanPath(conf) is " + Utilities.getPlanPath(job));
        Context.println("Map work:" + Utilities.getMapWork(job));
        // ctx = Utilities.getVectorizedRowBatchCtx(job);

        // int partitionColumnCount = ctx.getPartitionColumnCount();
        // partitionValues = new Object[partitionColumnCount];
        // VectorizedRowBatchCtx.getPartitionValues(ctx, job, (VineyardSplit)split,
        // partitionValues);
    }

    @Override
    public void close() throws IOException {
        if (client != null && client.connected()) {
            client.disconnect();
            Context.println("Bye, vineyard!");
        }
    }

    @Override
    public NullWritable createKey() {
        return NullWritable.get();
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
        return 0;
    }

    /**
     * Copy data from the Arrow RecordBatch to the VectorizedRowBatch.
     *
     * @param batch the VectorizedRowBatch to copy to
     */
    private void arrowToVectorizedRowBatch(VectorizedRowBatch batch) {
        // batch.numCols = recordBatch.getFieldVectors().size();
        int remaining = vectorSchemaRoot.getRowCount() - recordBatchInnerIndex;
        Context.println("partition column:" + batch.getPartitionColumnCount());
        Context.println("Data column count:" + batch.getDataColumnCount());

        /*
         * When we use SQL with condition to query data, the recordBatchSize may be large than VectorizedRowBatch.DEFAULT_SIZE,
         * which will cause ArrayIndexOutOfBoundsException. Because hive use a temporary array to store the data of
         * VectorizedRowBatch when process condition query, and the size of the array is VectorizedRowBatch.DEFAULT_SIZE
         * (VectorFilterOperator.java: 102).
         * So we need to limit the recordBatchSize to VectorizedRowBatch.DEFAULT_SIZE if the recordBatchSize is larger than
         * VectorizedRowBatch.DEFAULT_SIZE.
         */
        int recordBatchSize =
                remaining <= VectorizedRowBatch.DEFAULT_SIZE
                        ? remaining
                        : VectorizedRowBatch.DEFAULT_SIZE;
        batch.size = recordBatchSize;

        for (int i = 0; i < vectorSchemaRoot.getSchema().getFields().size(); i++) {
            Field field = vectorSchemaRoot.getSchema().getFields().get(i);
            if (field.getType().equals(Arrow.Type.Boolean)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BitVector bitVector = (BitVector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bitVector.get(k + recordBatchInnerIndex);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Int)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                IntVector intVector = (IntVector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = intVector.get(k + recordBatchInnerIndex);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Int64)) {
                LongColumnVector vector = new LongColumnVector(batch.size);
                BigIntVector bigIntVector =
                        (BigIntVector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = bigIntVector.get(k + recordBatchInnerIndex);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Float)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float4Vector float4Vector =
                        (Float4Vector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = float4Vector.get(k + recordBatchInnerIndex);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.Double)) {
                DoubleColumnVector vector = new DoubleColumnVector(batch.size);
                Float8Vector float8Vector =
                        (Float8Vector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.vector[k] = float8Vector.get(k + recordBatchInnerIndex);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.LargeVarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                LargeVarCharVector largeVarCharVector =
                        (LargeVarCharVector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    vector.setRef(
                            k,
                            largeVarCharVector.get(k + recordBatchInnerIndex),
                            0,
                            largeVarCharVector.get(k + recordBatchInnerIndex).length);
                }
                batch.cols[i] = vector;
            } else if (field.getType().equals(Arrow.Type.VarChar)) {
                BytesColumnVector vector = new BytesColumnVector(batch.size);
                vector.init();
                VarCharVector varCharVector =
                        (VarCharVector) vectorSchemaRoot.getFieldVectors().get(i);
                for (int k = 0; k < batch.size; k++) {
                    vector.isNull[k] = false;
                    vector.setRef(
                            k,
                            varCharVector.get(k + recordBatchInnerIndex),
                            0,
                            varCharVector.get(k + recordBatchInnerIndex).length);
                }
                batch.cols[i] = vector;
            } else {
                Context.println("array builder for type " + field.getType() + " is not supported");
                throw new UnsupportedOperationException(
                        "array builder for type " + field.getType() + " is not supported");
            }
        }
        recordBatchInnerIndex += recordBatchSize;
        if (recordBatchInnerIndex >= vectorSchemaRoot.getRowCount()) {
            recordBatchInnerIndex = 0;
            recordBatchIndex++;
            vectorSchemaRoot = null;
        }
    }

    @Override
    public boolean next(NullWritable key, VectorizedRowBatch value) throws IOException {
        Context.println("next");
        if (tableNameValid) {
            if (table == null) {
                try {
                    Context.println("Get objdect id:" + tableName);
                    ObjectID objectID = client.getName(tableName, false);
                    if (objectID != null) Context.println("get object done");
                    table =
                            (Table)
                                    ObjectFactory.getFactory()
                                            .resolve(client.getMetaData(objectID));
                } catch (Exception e) {
                    Context.println("Get objectID failed.");
                    return false;
                }
            }
            if (recordBatchIndex >= batchSize + batchStartIndex) {
                return false;
            }
            if (vectorSchemaRoot == null) {
                vectorSchemaRoot = table.getArrowBatch(recordBatchIndex);
            }
            if (vectorSchemaRoot == null) {
                Context.println("Get vector schema root failed.");
                return false;
            }

            if (addPartitionCols) {
                ctx.addPartitionColsToBatch(value, partitionValues);
                addPartitionCols = false;
            }
            this.arrowToVectorizedRowBatch(value);
            return true;
        }
        return false;
    }
}
