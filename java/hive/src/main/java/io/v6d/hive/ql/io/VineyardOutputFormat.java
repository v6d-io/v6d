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

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import lombok.val;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.arrow.vector.*;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.hive.ql.exec.FileSinkOperator;
import org.apache.hadoop.hive.ql.io.HiveOutputFormat;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.hive.common.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardOutputFormat<K extends NullWritable, V extends ArrowWrapperWritable>
        implements HiveOutputFormat<K, V> {
    private static Logger logger = LoggerFactory.getLogger(VineyardOutputFormat.class);

    @Override
    public FileSinkOperator.RecordWriter getHiveRecordWriter(
            JobConf jc,
            Path finalOutPath,
            Class<? extends Writable> valueClass,
            boolean isCompressed,
            Properties tableProperties,
            Progressable progress)
            throws IOException {
        return new SinkRecordWriter(
                jc, finalOutPath, valueClass, isCompressed, tableProperties, progress);
    }

    @Override
    public RecordWriter<K, V> getRecordWriter(
            FileSystem fileSystem, JobConf jobConf, String s, Progressable progressable)
            throws IOException {
        return new MapredRecordWriter<K, V>();
    }

    @Override
    public void checkOutputSpecs(FileSystem fileSystem, JobConf jobConf) throws IOException {}
}

class SinkRecordWriter implements FileSinkOperator.RecordWriter {
    private JobConf jc;
    private Path finalOutPath;
    private Properties tableProperties;
    private Progressable progress;
    private FileSystem fs;

    // vineyard
    private IPCClient client;
    private TableBuilder tableBuilder;
    private SchemaBuilder schemaBuilder;
    private List<RecordBatchBuilder> recordBatchBuilders;
    RecordBatchBuilder recordBatchBuilder;
    private String tableName;
    private List<String> partitions;
    private boolean hasData = false;

    public static final PathFilter VINEYARD_FILES_PATH_FILTER = new PathFilter() {
        @Override
        public boolean accept(Path p) {
            String name = p.getName();
            return name.startsWith("_task_tmp.") && (!name.substring(10, name.length()).startsWith("-"));
        }
    };

    private void getTableName(Properties tableProperties) {
        String location = tableProperties.getProperty("location");
        // int index = location.lastIndexOf("/");
        // tableName = location.substring(index + 1);
        System.out.println("finalOutPath : "+ finalOutPath.toString());

        // Get partition count
        String partition = tableProperties.getProperty("partition_columns.types");
        int index = -1;
        int partitionCount= 0;
        if (partition != null) {
            do {
                partitionCount++;
                index = partition.indexOf(":", index + 1);
            } while(index != -1);
        }
        System.out.println("Partition count:" + partitionCount);

        // Construct table name
        String path = finalOutPath.toString();
        path = path.substring(location.length(), path.length());
        String pathSplits[] = path.split("/");
        PathFilter vineyardPathFilter = VINEYARD_FILES_PATH_FILTER;
        PathFilter hiddernPathFilter = FileUtils.HIDDEN_FILES_PATH_FILTER;
        if (pathSplits.length == 0) {
            return;
        }
        tableName = location;
        for (int i = 1; i < pathSplits.length; i++) {
            if (pathSplits[i].length() > 0 && hiddernPathFilter.accept(new Path(pathSplits[i]))) {
                System.out.println("path split:" + pathSplits[i]);
                tableName += "/" + pathSplits[i];
            } else if (pathSplits[i].length() > 0 && vineyardPathFilter.accept(new Path(pathSplits[i]))) {
                System.out.println("path split:" + pathSplits[i].substring(10, pathSplits[i].length()));
                tableName += "/" + pathSplits[i].substring(10, pathSplits[i].length());
            }
        }
        tableName = tableName.replaceAll("/", "#");
        System.out.println("Table name:" + tableName);

        // Create temp file
        String tmpFilePath = finalOutPath.toString();
        tmpFilePath = tmpFilePath.substring(0, tmpFilePath.lastIndexOf("/"));
        System.out.println("out path:" + tmpFilePath);
        // File file = FileUtils.createTempFile(outPath, "vineyard", ".tmp");
        tmpFilePath = tmpFilePath.replaceAll("_task", "");
        Path tmpPath = new Path(tmpFilePath, "vineyard.tmp");
        try {
            fs = finalOutPath.getFileSystem(jc);
            System.out.println("tmp path:" + tmpPath.toString());
            FSDataOutputStream output = FileSystem.create(fs, tmpPath, new FsPermission("777"));
            if (output != null) {
                System.out.println("Create succeed!");
                output.write("test".getBytes(), 0, 4);
                output.close();
            }
            // System.in.read();
        } catch (Exception e) {
            System.out.println("Create failed!");
        }
        // index = location.length() + 1;
        // for (int i = 0; i < partitionCount; i++) {
        //     index = finalOutPath.toString().indexOf("/", index + 1);
        // }
        // tableName = finalOutPath.toString().substring(0, index);
        // tableName = tableName.replace('/', '#');
        // System.out.println("Table name:" + tableName);
        // System.out.println("fina path:" + finalOutPath.toString());

    }

    @lombok.SneakyThrows
    public SinkRecordWriter(
            JobConf jc,
            Path finalOutPath,
            Class<? extends Writable> valueClass,
            boolean isCompressed,
            Properties tableProperties,
            Progressable progress) {
        this.jc = jc;
        if (!ArrowWrapperWritable.class.isAssignableFrom(valueClass)) {
            throw new VineyardException.Invalid("value class must be ArrowWrapperWritable");
        }
        if (isCompressed) {
            throw new VineyardException.Invalid("compressed output is not supported");
        }
        this.finalOutPath = finalOutPath;
        this.tableProperties = tableProperties;
        this.progress = progress;

        for (Object key : tableProperties.keySet()) {
            System.out.printf("table property: %s, %s\n", key, tableProperties.getProperty((String) key));
        }
        getTableName(tableProperties);

        // connect to vineyard
        if (client == null) {
            // TBD: get vineyard socket path from table properties
            client = new IPCClient("/tmp/vineyard/vineyard.sock");
        }
        if (client == null || !client.connected()) {
            throw new VineyardException.Invalid("failed to connect to vineyard");
        } else {
            System.out.printf("Connected to vineyard succeed!\n");
            System.out.printf("Hello vineyard!\n");
        }

        recordBatchBuilders = new ArrayList<RecordBatchBuilder>();
        Arrow.instantiate();
    }

    @Override
    public void write(Writable w) throws IOException {
        System.out.println("write");
        if (w == null) {
            System.out.println("w is null");
            return;
        }
        ArrowWrapperWritable arrowWrapperWritable = (ArrowWrapperWritable) w;
        VectorSchemaRoot root = arrowWrapperWritable.getVectorSchemaRoot();
        if (root == null || root.getRowCount() == 0) {
            return;
        }
        System.out.println("Row count:" + root.getRowCount() + " Field size:" + root.getSchema().getFields().size() + " vector size:" + root.getFieldVectors().size());
        hasData = true;
        fillRecordBatchBuilder(root);
    }
    
    // check if the table is already created.
    // if not, create a new table.
    // if yes, append the data to the table.(Get from vineyard, and seal it in a new table) 
    @Override
    public void close(boolean abort) throws IOException {
        System.out.println("close");
        System.out.println("table name:" + tableName);
        Table oldTable = null;
        System.out.println("has data:" + hasData);
        if (schemaBuilder == null) {
            System.out.println("No data to write.");
            client.disconnect();
            System.out.println("Bye, vineyard!");
            return;
        }
        tableBuilder = new TableBuilder(client, schemaBuilder);
        try {
            ObjectID objectID = client.getName(tableName, false);
            if (objectID == null) {
                System.out.println("Table not exist.");
            } else {
                oldTable = (Table) ObjectFactory.getFactory().resolve(client.getMetaData(objectID));
            }
        } catch (Exception e) {
            System.out.println("Get table id failed");
        }

        if (oldTable != null) {
            for (int i = 0; i < oldTable.getBatches().size(); i++) {
                tableBuilder.addBatch(oldTable.getBatches().get(i));
            }
        }

        for (int i = 0; i < recordBatchBuilders.size(); i++) {
            tableBuilder.addBatch(recordBatchBuilders.get(i));
        }
        try {
            ObjectMeta meta = tableBuilder.seal(client);
            System.out.println("Table id in vineyard:" + meta.getId().value());
            client.persist(meta.getId());
            System.out.println("Table persisted, name:" + tableName);
            client.putName(meta.getId(), tableName);
        } catch (Exception e) {
            throw new IOException("Seal TableBuilder failed");
        }
        client.disconnect();

        System.out.println("Bye, vineyard!");
    }

    private void fillRecordBatchBuilder(VectorSchemaRoot root) throws IOException{
        org.apache.arrow.vector.types.pojo.Schema schema = root.getSchema();
        schemaBuilder = SchemaBuilder.fromSchema(schema);
        try {
            // TBD : more clear error message.
            recordBatchBuilder = new RecordBatchBuilder(client, schema, root.getRowCount());
            fillColumns(root, schema);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            throw new IOException("Add field failed");
        }
        recordBatchBuilders.add(recordBatchBuilder);
    }

    private void fillColumns(VectorSchemaRoot root, Schema schema)
        throws VineyardException {
        for (int i = 0; i < schema.getFields().size(); i++) {
            val column = recordBatchBuilder.getColumnBuilder(i);
            Field field = schema.getFields().get(i);
            if (field.getType().equals(Arrow.Type.Boolean)) {
                BitVector vector = (BitVector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    if (vector.get(j) != 0) {
                        column.setBoolean(j, true);
                    } else {
                        column.setBoolean(j, false);
                    }
                }
            } else if (field.getType().equals(Arrow.Type.Int)) {
                IntVector vector= (IntVector) root.getFieldVectors().get(i);
                // System.out.println("row count:" + root.getRowCount());
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setInt(j, vector.get(j));
                    // System.out.println("int value:" + vector.get(j));
                }
            } else if (field.getType().equals(Arrow.Type.Int64)) {
                BigIntVector vector = (BigIntVector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setLong(j, vector.get(j));
                }
            } else if (field.getType().equals(Arrow.Type.Float)) {
                Float4Vector vector = (Float4Vector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setFloat(j, vector.get(j));
                }
            } else if (field.getType().equals(Arrow.Type.Double)) {
                Float8Vector vector = (Float8Vector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setDouble(j, vector.get(j));
                } 
            } else if (field.getType().equals(Arrow.Type.LargeVarChar)) {
                LargeVarCharVector vector = (LargeVarCharVector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setUTF8String(j, vector.getObject(j));
                } 
            } else if (field.getType().equals(Arrow.Type.VarChar)) {
                VarCharVector vector = (VarCharVector) root.getFieldVectors().get(i);
                for (int j = 0; j < root.getRowCount(); j++) {
                    column.setUTF8String(j, vector.getObject(j));
                } 
            } else {
                throw new VineyardException.NotImplemented(
                        "array builder for type " + field.getType() + " is not supported");
            }
        }
    }
}

class MapredRecordWriter<K extends NullWritable, V extends ArrowWrapperWritable>
        implements RecordWriter<K, V> {
    MapredRecordWriter() throws IOException {
        System.out.printf("creating vineyard record writer\n");
        throw new RuntimeException("mapred record writter: unimplemented");
    }

    @Override
    public void write(K k, V v) throws IOException {
        System.out.printf("write: k = %s, v = %s\n", k, v);
    }

    @Override
    public void close(Reporter reporter) throws IOException {
        System.out.printf("closing\n");
    }
}

// file:/opt/hive/data/warehouse/hive_dynamic_partition_test6/.hive-staging_hive_2023-07-19_10-49-12_537_8312530081206971822-1/_task_tmp.-ext-10000/year=2017/_tmp.000000_0
// file:/opt/hive/data/warehouse/hive_dynamic_partition_test6/.hive-staging_hive_2023-07-19_10-49-12_537_8312530081206971822-1/-ext-10000

// file:/opt/hive/data/warehouse/hive_dynamic_partition_test6/year=2018/.hive-staging_hive_2023-07-19_10-52-50_144_3570501016920325767-1/_task_tmp.-ext-10000/_tmp.000000_0
// file:/opt/hive/data/warehouse/hive_dynamic_partition_test6/year=2018/.hive-staging_hive_2023-07-19_10-52-50_144_3570501016920325767-1/-ext-10000

//file:/opt/hive/data/warehouse/hive_dynamic_partition_test6/.hive-staging_hive_2023-07-19_11-15-58_433_128099267613906011-1/-ext-10000
//file:/opt/hive/data/warehouse/hive_dynamic_partition_test7/.hive-staging_hive_2023-07-19_11-14-48_835_1857436151368976840-1/-ext-10000