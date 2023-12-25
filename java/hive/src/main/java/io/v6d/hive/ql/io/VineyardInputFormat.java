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
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException.ObjectNotExists;
import io.v6d.modules.basic.arrow.*;
import io.v6d.modules.basic.arrow.util.ObjectResolver;
import io.v6d.modules.basic.columnar.ColumnarData;
import io.v6d.modules.basic.filesystem.VineyardFileUtils;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import lombok.val;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.common.FileUtils;
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

public class VineyardInputFormat extends HiveInputFormat<NullWritable, RecordWrapperWritable>
        implements VectorizedInputFormatInterface {
    @Override
    public RecordReader<NullWritable, RecordWrapperWritable> getRecordReader(
            InputSplit genericSplit, JobConf job, Reporter reporter) throws IOException {
        reporter.setStatus(genericSplit.toString());
        return new VineyardRecordReader(job, (VineyardSplit) ((VineyardSplit) genericSplit));
    }

    @Override
    public InputSplit[] getSplits(JobConf job, int numSplits) throws IOException {
        Path paths[] = FileInputFormat.getInputPaths(job);

        val client = Context.getClient();
        // split table by paths
        List<VineyardSplit> splits = new ArrayList<>();
        Arrow.instantiate();

        for (int i = 0; i < paths.length; i++) {
            // Construct table name.
            Path path = paths[i];

            // get object id from vineyard filesystem
            FileSystem fs = path.getFileSystem(job);
            FileStatus[] tableStatus = fs.listStatus(path, FileUtils.HIDDEN_FILES_PATH_FILTER);
            if (tableStatus.length == 0) {
                continue;
            }
            Queue<FileStatus[]> dirStatus = new LinkedList<>();
            dirStatus.add(tableStatus);

            // Maybe there exists more than one table file.
            VineyardFileUtils.printAllFiles(Context.getClient());
            while (!dirStatus.isEmpty()) {
                FileStatus[] status = dirStatus.poll();
                for (int j = 0; j < status.length; j++) {
                    int batchesPerFile = 0;
                    if (status[j].isDirectory()) {
                        dirStatus.add(
                                fs.listStatus(
                                        status[j].getPath(), FileUtils.HIDDEN_FILES_PATH_FILTER));
                        continue;
                    }
                    Path tableFilePath = status[j].getPath();
                    try (FSDataInputStream in = fs.open(tableFilePath)) {
                        FileStatus fileStatus = fs.getFileStatus(tableFilePath);
                        byte[] buffer = new byte[(int) fileStatus.getLen()];
                        int len = in.read(buffer, 0, (int) fileStatus.getLen());
                        // Here must check with the condition of len <= 0, rather than len == -1.
                        // Because Spark will create an empty file, which will cause the len == 0.
                        if (len <= 0) {
                            continue;
                        }

                        // FIXME: Shoule be only one object id in each file.
                        Set<String> hostSet = new HashSet<>();
                        String[] objectIDs =
                                new String(buffer, StandardCharsets.US_ASCII).split("\n");
                        for (val objectID : objectIDs) {
                            ObjectMeta tableMeta;
                            Table table;
                            try {
                                ObjectID tableID = ObjectID.fromString(objectID);
                                tableMeta = client.getMetaData(tableID, false);
                            } catch (ObjectNotExists | NumberFormatException e) {
                                // Skip some invalid file.
                                Context.println(
                                        "Skipping invalid file: "
                                                + tableFilePath
                                                + ", content: "
                                                + new String(buffer, StandardCharsets.US_ASCII));
                                break;
                            }
                            table = (Table) ObjectFactory.getFactory().resolve(tableMeta);
                            batchesPerFile += table.getBatchNum();
                            for (String host :
                                    table.getHostsOfRecordBatches(0, table.getBatchNum())) {
                                hostSet.add(host);
                            }
                        }

                        splits.add(
                                new VineyardSplit(
                                        status[j].getPath(),
                                        0,
                                        batchesPerFile,
                                        hostSet.toArray(new String[hostSet.size()]),
                                        hostSet.toArray(new String[hostSet.size()])));
                    }
                }
            }
        }
        return splits.toArray(new VineyardSplit[splits.size()]);
    }
}

class VineyardRecordReader implements RecordReader<NullWritable, RecordWrapperWritable> {
    private static Logger logger = LoggerFactory.getLogger(VineyardRecordReader.class);

    private RecordBatch[] batches;
    private int recordBatchIndex = -1;
    private int recordBatchInnerIndex = 0;
    private Schema schema;
    private VectorSchemaRoot batch;
    private ColumnarData[] columns;
    private Stopwatch watch = StopwatchContext.createUnstarted();
    private ObjectResolver resolver;

    private long recordTotal = 0;
    private long recordConsumed = 0;

    VineyardRecordReader(JobConf job, VineyardSplit split) throws IOException {
        this.batches = new RecordBatch[(int) split.getLength()];
        this.recordBatchIndex = 0;

        Path path = split.getPath();

        FileSystem fs = path.getFileSystem(job);
        FileStatus tableStatus = fs.getFileStatus(path);

        val client = Context.getClient();
        Arrow.instantiate();

        resolver = new HiveTypeResolver();

        if (tableStatus.isDirectory()) {
            throw new IOException("Path of table is a dir!");
        }

        try (FSDataInputStream in = fs.open(path)) {
            byte[] buffer = new byte[(int) tableStatus.getLen()];
            int len = in.read(buffer, 0, (int) tableStatus.getLen());
            if (len <= 0) {
                return;
            }

            // getSplits will ensure that the file referenced by path is a valid table.
            String[] objectIDs = new String(buffer, StandardCharsets.US_ASCII).split("\n");
            ObjectID tableID = ObjectID.fromString(objectIDs[0]);
            ObjectMeta tableMeta = client.getMetaData(tableID, false);
            Table table = (Table) ObjectFactory.getFactory().resolve(tableMeta);

            recordTotal = split.getLength();
            for (int i = (int) split.getStart(); i < split.getLength() + split.getStart(); i++) {
                val batch = table.getBatch(i);
                batch.setResolver(resolver);
                this.batches[this.recordBatchIndex++] = batch;
            }
            schema = table.getSchema().getSchema();
            // reset to the beginning
            this.recordBatchIndex = -1;
            this.recordBatchInnerIndex = 0;
        }
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
    public RecordWrapperWritable createValue() {
        return new RecordWrapperWritable(schema);
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
    public boolean next(NullWritable key, RecordWrapperWritable value) throws IOException {
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
