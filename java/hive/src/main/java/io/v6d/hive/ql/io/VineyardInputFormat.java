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

import com.google.common.base.Stopwatch;
import com.google.common.base.StopwatchContext;
import io.v6d.core.client.Context;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.*;
import io.v6d.modules.basic.columnar.ColumnarData;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import lombok.val;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.vector.VectorizedInputFormatInterface;
import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardInputFormat extends HiveInputFormat<NullWritable, RowWritable>
        implements VectorizedInputFormatInterface {
    @Override
    public RecordReader<NullWritable, RowWritable> getRecordReader(
            InputSplit genericSplit, JobConf job, Reporter reporter) throws IOException {
        reporter.setStatus(genericSplit.toString());
        return new VineyardRecordReader(job, (VineyardSplit) genericSplit);
    }

    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
        Path paths[] = FileInputFormat.getInputPaths(job);

        val client = Context.getClient();
        val splits = new VineyardSplit[paths.length];
        Arrow.instantiate();

        for (int i = 0; i < paths.length; i++) {
            // Construct table name.
            Path path = paths[i];
            String tableName = path.toString();

            // get object id from vineyard filesystem
            FileSystem fs = path.getFileSystem(job);
            FileStatus[] status = fs.listStatus(path);
            if (status.length == 0) {
                throw new VineyardException.ObjectNotExists("Table not found: " + tableName);
            }

            // Maybe there exists more than one table file.
            long numBatches = 0;
            for (int j = 0; j < status.length; j++) {
                Path tableFilePath = status[j].getPath();
                FSDataInputStream in = fs.open(tableFilePath);
                FileStatus fileStatus = fs.getFileStatus(tableFilePath);
                byte[] buffer = new byte[(int) fileStatus.getLen()];
                int len = in.read(buffer, 0, (int) fileStatus.getLen());
                /*
                 * Here must check with the condition of len <= 0, rather than len == -1.
                 * Because Spark will create a empty file, which will cause the len == 0.
                 */
                if (len <= 0) {
                    continue;
                }
                String[] objectIDs = new String(buffer, StandardCharsets.UTF_8).split("\n");
                for (val objectID : objectIDs) {
                    try {
                        ObjectID tableID = ObjectID.fromString(objectID);
                        Table table =
                                (Table)
                                        ObjectFactory.getFactory()
                                                .resolve(client.getMetaData(tableID));
                        numBatches += table.getBatches().size();
                    } catch (Exception e) {
                        // Skip some invalid file.
                        Context.println("Skip invalid file:" + tableFilePath);
                        Context.println(
                                "File content:" + new String(buffer, StandardCharsets.UTF_8));
                        break;
                    }
                }
            }
            // TODO: would generating a split for each record batch be better?
            splits[i] = new VineyardSplit(path, 0, numBatches, job);
        }
        return splits;
    }
}

class VineyardRecordReader implements RecordReader<NullWritable, RowWritable> {
    private static Logger logger = LoggerFactory.getLogger(VineyardRecordReader.class);

    private RecordBatch[] batches;
    private int recordBatchIndex = -1;
    private int recordBatchInnerIndex = 0;
    private Schema schema;
    private VectorSchemaRoot batch;
    private ColumnarData[] columns;
    private Stopwatch watch = StopwatchContext.createUnstarted();

    private long recordTotal = 0;
    private long recordConsumed = 0;

    VineyardRecordReader(JobConf job, VineyardSplit split) throws IOException {
        Path path = split.getPath();
        String tableName = path.toString();

        FileSystem fs = path.getFileSystem(job);
        FileStatus[] status = fs.listStatus(path);
        if (status.length == 0) {
            throw new VineyardException.ObjectNotExists("Table not found: " + tableName);
        }

        val client = Context.getClient();
        Arrow.instantiate();

        this.batches = new RecordBatch[(int) split.getLength()];
        this.recordBatchIndex = 0;

        for (int j = 0; j < status.length; j++) {
            Path tableFilePath = status[j].getPath();
            FSDataInputStream in = fs.open(tableFilePath);
            FileStatus fileStatus = fs.getFileStatus(tableFilePath);
            byte[] buffer = new byte[(int) fileStatus.getLen()];
            int len = in.read(buffer, 0, (int) fileStatus.getLen());
            if (len <= 0) {
                continue;
            }
            String[] objectIDs = new String(buffer, StandardCharsets.UTF_8).split("\n");
            for (val objectID : objectIDs) {
                try {
                    ObjectID tableID = ObjectID.fromString(objectID);
                    Table table =
                            (Table) ObjectFactory.getFactory().resolve(client.getMetaData(tableID));
                    for (val batch : table.getBatches()) {
                        recordTotal += batch.getRowCount();
                        this.batches[this.recordBatchIndex++] = batch;
                        schema = table.getSchema().getSchema();
                    }
                } catch (Exception e) {
                    // Skip some invalid file.
                    Context.println("Skip invalid file:" + tableFilePath);
                    Context.println("File content:" + new String(buffer, StandardCharsets.UTF_8));
                    break;
                }
            }
        }
        // reset to the beginning
        this.recordBatchIndex = -1;
        this.recordBatchInnerIndex = 0;
    }

    @Override
    public void close() throws IOException {
        // do nothing
    }

    @Override
    public NullWritable createKey() {
        return null;
    }

    @Override
    public RowWritable createValue() {
        return new RowWritable(schema);
    }

    /** N.B.: this method must be accurate and is important for selection performance. */
    @Override
    public long getPos() throws IOException {
        return recordConsumed;
    }

    @Override
    public float getProgress() throws IOException {
        return ((float) recordConsumed) / recordTotal;
    }

    @Override
    public boolean next(NullWritable key, RowWritable value) throws IOException {
        watch.start();
        // initialize the current batch
        while (batch == null || recordBatchInnerIndex >= batch.getRowCount()) {
            if (recordBatchIndex + 1 >= batches.length) {
                return false;
            }
            recordBatchIndex++;
            if (recordBatchIndex >= batches.length) {
                return false;
            }
            batch = batches[recordBatchIndex].getBatch();
            columns = batches[recordBatchIndex].columar();
            recordBatchInnerIndex = 0;
            Context.println(
                    "record batch length: "
                            + batch.getRowCount()
                            + " time usage util now: "
                            + watch);
        }

        // update the value
        value.setValues(columns, recordBatchInnerIndex);

        // move cursor to next record
        recordBatchInnerIndex++;
        recordConsumed++;
        watch.stop();
        return true;
    }
}
